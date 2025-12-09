# inference_final.py
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
from tqdm import tqdm
import warnings
import time

warnings.filterwarnings('ignore')

# Configuración
SETTINGS = {
    "TRAIN_RAW_DATA_PATH": "./data/de_train.parquet",
    "TEST_RAW_DATA_PATH": "./data/id_map.csv",
    "TRAIN_DATA_AUG_DIR": "./features/"
}

def load_test_data():
    """Carga datos de test y características"""
    print("📥 Cargando datos de test...")
    
    # Cargar ID map
    id_map = pd.read_csv(SETTINGS["TEST_RAW_DATA_PATH"])
    print(f"   Muestras en test: {len(id_map)}")
    
    # Cargar características
    print("\n🔄 Cargando características...")
    aug_dir = SETTINGS["TRAIN_DATA_AUG_DIR"]
    
    try:
        # One-hot encodings
        oh_cell = pd.read_csv(f"{aug_dir}one_hot_cell_type_test.csv")
        oh_sm = pd.read_csv(f"{aug_dir}one_hot_sm_name_test.csv")
        
        # ChemBERTa features
        chem_feats = pd.read_csv(f"{aug_dir}ChemBERTa_test.csv")
        
        # Combinar
        features = pd.concat([oh_cell, oh_sm, chem_feats], axis=1)
        
        # Rellenar NaN si los hay
        if features.isnull().any().any():
            print(f"   ⚠️ Rellenando {features.isnull().sum().sum()} valores NaN con 0")
            features = features.fillna(0)
        
        print(f"   ✅ Características cargadas: {features.shape}")
        
    except Exception as e:
        print(f"   ❌ Error cargando características: {e}")
        return None, None
    
    return id_map, features

def load_models_and_metadata():
    """Carga modelos y metadatos"""
    print("\n🔄 Cargando modelos y metadatos...")
    
    # Cargar metadatos
    try:
        with open("models_cuda/metadata_cuda.json", "r") as f:
            metadata = json.load(f)
        
        gene_names = metadata['target_cols']
        n_genes = metadata['n_genes']
        n_folds = metadata['n_folds']
        
        print(f"   ✅ Metadatos cargados:")
        print(f"     - Genes: {n_genes}")
        print(f"     - Folds: {n_folds}")
        print(f"     - CV Score: {metadata['cv_score']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error cargando metadatos: {e}")
        return None, None, None
    
    # Cargar modelos desde cada fold
    models = {}
    
    for fold_num in range(n_folds):
        fold_dir = f"models_cuda/fold_{fold_num}"
        
        if not os.path.exists(fold_dir):
            print(f"   ⚠️ {fold_dir} no existe")
            continue
        
        # Listar archivos de modelo
        model_files = [f for f in os.listdir(fold_dir) 
                      if f.endswith('.txt') and f.startswith('gene_')]
        
        if not model_files:
            print(f"   ⚠️ No hay modelos en {fold_dir}")
            continue
        
        # Ordenar numéricamente
        model_files = sorted(model_files, 
                           key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        print(f"\n   📂 Cargando {fold_dir}...")
        fold_models = []
        
        for model_file in tqdm(model_files, desc=f"     Modelos", leave=False):
            model_path = os.path.join(fold_dir, model_file)
            model = lgb.Booster(model_file=model_path)
            fold_models.append(model)
        
        models[f'fold_{fold_num}'] = fold_models
        print(f"     ✓ {len(fold_models)} modelos cargados")
    
    return models, gene_names, metadata

def verify_consistency(models, gene_names, metadata):
    """Verifica que todo sea consistente"""
    print("\n🔍 Verificando consistencia...")
    
    if not models:
        print("   ❌ No hay modelos cargados")
        return False
    
    # Número de folds
    n_folds_loaded = len(models)
    n_folds_expected = metadata['n_folds']
    
    if n_folds_loaded != n_folds_expected:
        print(f"   ⚠️ Folds cargados ({n_folds_loaded}) != esperados ({n_folds_expected})")
    
    # Número de modelos por fold
    first_fold = list(models.keys())[0]
    n_models_per_fold = len(models[first_fold])
    n_genes_expected = metadata['n_genes']
    
    print(f"   Folds cargados: {n_folds_loaded}")
    print(f"   Modelos por fold: {n_models_per_fold}")
    print(f"   Genes esperados: {n_genes_expected}")
    
    if n_models_per_fold != n_genes_expected:
        print(f"   ⚠️ Modelos por fold ({n_models_per_fold}) != genes ({n_genes_expected})")
        print(f"   Ajustando lista de genes...")
        
        if n_models_per_fold < len(gene_names):
            gene_names = gene_names[:n_models_per_fold]
            print(f"   Recortando genes a {n_models_per_fold}")
        else:
            # Esto no debería pasar, pero por si acaso
            print(f"   ❌ ERROR: Más modelos que genes")
            return False
    
    # Verificar que todos los folds tengan el mismo número de modelos
    for fold_name, fold_models in models.items():
        if len(fold_models) != n_models_per_fold:
            print(f"   ⚠️ {fold_name} tiene {len(fold_models)} modelos (esperaba {n_models_per_fold})")
    
    print("   ✅ Consistencia verificada")
    return True, gene_names

def predict_with_all_folds(models, features):
    """Genera predicciones promediando todos los folds"""
    if not models:
        return None
    
    first_fold = list(models.keys())[0]
    n_samples = features.shape[0]
    n_genes = len(models[first_fold])
    n_folds = len(models)
    
    print(f"\n🔮 Generando predicciones...")
    print(f"   Muestras: {n_samples}")
    print(f"   Genes: {n_genes}")
    print(f"   Folds: {n_folds}")
    
    # Inicializar array para predicciones de todos los folds
    all_predictions = np.zeros((n_folds, n_samples, n_genes))
    
    start_time = time.time()
    
    # Procesar cada fold
    for fold_idx, (fold_name, fold_models) in enumerate(models.items()):
        print(f"\n   📊 Procesando {fold_name}...")
        
        fold_predictions = np.zeros((n_samples, n_genes))
        
        # Predecir todos los genes
        for gene_idx in tqdm(range(n_genes), desc="     Genes", leave=False):
            model = fold_models[gene_idx]
            pred = model.predict(features)
            fold_predictions[:, gene_idx] = pred
        
        all_predictions[fold_idx] = fold_predictions
        
        # Tiempo parcial
        partial_time = time.time() - start_time
        print(f"     ✓ Completado en {partial_time:.1f}s")
    
    # Promediar predicciones de todos los folds
    print(f"\n   📈 Promediando {n_folds} folds...")
    final_predictions = np.mean(all_predictions, axis=0)
    
    total_time = time.time() - start_time
    print(f"\n   ✅ Predicciones completadas")
    print(f"     Tiempo total: {total_time:.1f}s")
    print(f"     Velocidad: {(n_genes * n_folds) / total_time:.1f} predicciones/s")
    
    return final_predictions

def create_submission_file(predictions, id_map, gene_names, output_filename="submission_final.csv"):
    """Crea el archivo de submission para Kaggle"""
    print(f"\n💾 Creando archivo de submission...")
    
    # Verificar dimensiones
    n_samples, n_genes_pred = predictions.shape
    n_gene_names = len(gene_names)
    
    print(f"   Predicciones: {n_samples} × {n_genes_pred}")
    print(f"   Nombres de genes: {n_gene_names}")
    
    # Asegurar que coincidan
    if n_genes_pred != n_gene_names:
        print(f"   ⚠️ Ajustando: predicciones={n_genes_pred}, nombres={n_gene_names}")
        if n_genes_pred < n_gene_names:
            gene_names = gene_names[:n_genes_pred]
        else:
            # Esto no debería pasar
            gene_names = gene_names + [f"gene_extra_{i}" for i in range(n_genes_pred - n_gene_names)]
    
    # Crear DataFrame en formato ancho
    submission_wide = pd.DataFrame(predictions, columns=gene_names)
    submission_wide.insert(0, 'id', id_map['id'].values)
    
    print(f"   DataFrame ancho: {submission_wide.shape}")
    
    # Convertir a formato largo (para Kaggle)
    submission_long = submission_wide.melt(
        id_vars=['id'],
        var_name='gene',
        value_name='target'
    )
    
    print(f"   DataFrame largo: {submission_long.shape}")
    
    # Guardar
    submission_long.to_csv(output_filename, index=False)
    
    # Estadísticas
    stats = {
        'min': float(predictions.min()),
        'max': float(predictions.max()),
        'mean': float(predictions.mean()),
        'std': float(predictions.std()),
        'percentile_1': float(np.percentile(predictions, 1)),
        'percentile_99': float(np.percentile(predictions, 99))
    }
    
    print(f"\n📊 Estadísticas de predicciones:")
    print(f"   Rango: [{stats['min']:.6f}, {stats['max']:.6f}]")
    print(f"   Media: {stats['mean']:.6f}")
    print(f"   Desviación estándar: {stats['std']:.6f}")
    print(f"   Percentil 1%: {stats['percentile_1']:.6f}")
    print(f"   Percentil 99%: {stats['percentile_99']:.6f}")
    
    # Guardar estadísticas
    with open("submission_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    return submission_long, stats

def main():
    print("="*80)
    print("🎯 INFERENCIA FINAL - PREDICCIÓN CON 18,211 GENES")
    print("="*80)
    
    # 1. Cargar datos de test
    print("\n1. Cargando datos de test...")
    id_map, features = load_test_data()
    
    if id_map is None or features is None:
        print("❌ Error cargando datos de test")
        return
    
    # 2. Cargar modelos y metadatos
    print("\n2. Cargando modelos entrenados...")
    models, gene_names, metadata = load_models_and_metadata()
    
    if models is None or gene_names is None:
        print("❌ Error cargando modelos")
        return
    
    # 3. Verificar consistencia
    print("\n3. Verificando consistencia...")
    result, gene_names = verify_consistency(models, gene_names, metadata)
    if not result:
        print("❌ Error de consistencia")
        return
    
    # 4. Generar predicciones
    print("\n4. Generando predicciones (esto puede tomar tiempo)...")
    predictions = predict_with_all_folds(models, features)
    
    if predictions is None:
        print("❌ Error generando predicciones")
        return
    
    # 5. Crear archivo de submission
    print("\n5. Creando archivo final...")
    submission, stats = create_submission_file(predictions, id_map, gene_names)
    
    print(f"\n{'='*80}")
    print("✅ ¡INFERENCIA COMPLETADA EXITOSAMENTE!")
    print(f"{'='*80}")
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    print(f"   1. submission_final.csv - Para Kaggle ({os.path.getsize('submission_final.csv') / 1024 / 1024:.1f} MB)")
    print(f"   2. submission_stats.json - Estadísticas")

    print(f"\n RESUMEN FINAL:")
    print(f"   • Muestras predichas: {len(id_map)}")
    print(f"   • Genes predichos: {predictions.shape[1]}")
    print(f"   • Filas en submission: {len(submission):,}")
    print(f"   • RMSE del modelo: {metadata['cv_score']:.4f}")
    
    print(f"\n PARA KAGGLE:")
    print(f"   Sube el archivo: submission_final.csv")
    
    # Mostrar ejemplo
    print(f"\n👀 EJEMPLO (primeras 5 filas):")
    print(submission.head(10).to_string(index=False))
    
    # Crear archivo de confirmación
    with open("INFERENCE_COMPLETE.txt", "w") as f:
        f.write("INFERENCIA COMPLETADA EXITOSAMENTE\n")
        f.write("="*40 + "\n")
        f.write(f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Archivo: submission_final.csv\n")
        f.write(f"Muestras: {len(id_map)}\n")
        f.write(f"Genes: {predictions.shape[1]}\n")
        f.write(f"Filas totales: {len(submission)}\n")
        f.write(f"RMSE modelo: {metadata['cv_score']:.4f}\n")
        f.write(f"Tamaño archivo: {os.path.getsize('submission_final.csv') / 1024 / 1024:.1f} MB\n")

if __name__ == "__main__":
    main()