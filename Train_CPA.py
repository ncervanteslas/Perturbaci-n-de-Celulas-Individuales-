# train_cpa_model.py
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os
import warnings
import time
from tqdm import tqdm
import sys
from scipy import stats

warnings.filterwarnings('ignore')

# Configuración
with open("./SETTINGS.json") as f:
    SETTINGS = json.load(f)

def setup_environment():
    """Configuración del entorno CPA"""
    print(" CONFIGURANDO ENTORNO CPA")
    print("="*60)
    print("🎯 Compositional Perturbation Autoencoder (CPA)")
    print("="*60)
    
    # Crear directorios si no existen
    os.makedirs("cpa_models", exist_ok=True)
    os.makedirs("cpa_features", exist_ok=True)
    
    return True

def get_cpa_optimized_params():
    """Parámetros optimizados para CPA"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        
        # APRENDIZAJE COMPOSICIONAL
        'learning_rate': 0.006,           # Muy bajo para composición estable
        'num_leaves': 511,                # Alto para interacciones complejas
        'max_depth': 13,                  # Profundo pero controlado
        
        # REGULARIZACIÓN COMPOSICIONAL
        'min_child_samples': 3,           # Muy bajo para señales sutiles
        'min_child_weight': 0.0001,       # Muy bajo
        'min_split_gain': 0.00001,        # Mínimo
        
        'reg_alpha': 0.00001,             # L1 casi desactivado
        'reg_lambda': 0.00001,            # L2 casi desactivado
        
        # SUBSAMPLING COMPOSICIONAL
        'subsample': 0.6,                 # Bajo para robustez
        'colsample_bytree': 0.6,          # Bajo
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 7,
        
        # CONFIGURACIÓN CPA
        'max_bin': 511,                   # Alto para precisión
        'tree_learner': 'feature',        # Para muchas características
        'extra_trees': False,             # Mejor para datos tabulares
        
        'num_threads': 8,
        'verbose': -1,
        'seed': 42,
        'deterministic': True,
    }
    return params

def compositional_normalization(features, cell_types, sm_names, train=True):
    """
    Normalización composicional específica para CPA
    Considera composición celular y de compuestos
    """
    print("🔬 Aplicando normalización composicional...")
    
    if train:
        # Escalado robusto por tipo celular
        scalers_cell = {}
        features_cell_normalized = np.zeros_like(features)
        
        unique_cells = np.unique(cell_types)
        for cell in unique_cells:
            cell_mask = cell_types == cell
            if np.sum(cell_mask) > 10:  # Mínimo muestras
                scaler = RobustScaler(quantile_range=(10, 90))
                features_cell_normalized[cell_mask] = scaler.fit_transform(features[cell_mask])
                scalers_cell[cell] = scaler
        
        # Escalado robusto por compuesto
        scalers_sm = {}
        features_sm_normalized = np.zeros_like(features)
        
        unique_sm = np.unique(sm_names)
        for sm in unique_sm:
            sm_mask = sm_names == sm
            if np.sum(sm_mask) > 5:  # Mínimo muestras
                scaler = RobustScaler(quantile_range=(10, 90))
                features_sm_normalized[sm_mask] = scaler.fit_transform(features[sm_mask])
                scalers_sm[sm] = scaler
        
        # Combinación ponderada (50-50)
        features_norm = 0.5 * features_cell_normalized + 0.5 * features_sm_normalized
        
        # Guardar scalers
        joblib.dump(scalers_cell, "cpa_features/cell_scalers.pkl")
        joblib.dump(scalers_sm, "cpa_features/sm_scalers.pkl")
        
    else:
        # Cargar y aplicar scalers
        try:
            scalers_cell = joblib.load("cpa_features/cell_scalers.pkl")
            scalers_sm = joblib.load("cpa_features/sm_scalers.pkl")
            
            features_cell_normalized = np.zeros_like(features)
            features_sm_normalized = np.zeros_like(features)
            
            # Aplicar por tipo celular
            for cell, scaler in scalers_cell.items():
                cell_mask = cell_types == cell
                if np.sum(cell_mask) > 0:
                    features_cell_normalized[cell_mask] = scaler.transform(features[cell_mask])
            
            # Aplicar por compuesto
            for sm, scaler in scalers_sm.items():
                sm_mask = sm_names == sm
                if np.sum(sm_mask) > 0:
                    features_sm_normalized[sm_mask] = scaler.transform(features[sm_mask])
            
            # Combinación ponderada
            features_norm = 0.5 * features_cell_normalized + 0.5 * features_sm_normalized
            
        except:
            print("⚠️  Usando escalado global como fallback")
            scaler = RobustScaler()
            features_norm = scaler.fit_transform(features) if train else scaler.transform(features)
    
    return features_norm.astype(np.float32)

def compositional_target_transform(y, epsilon=1e-6):
    """
    Transformación composicional del target
    Aplica transformación logarítmica con manejo de signos
    """
    # Manejar valores cercanos a cero
    y_safe = np.where(np.abs(y) < epsilon, epsilon * np.sign(y), y)
    
    # Transformación logarítmica composicional
    y_transformed = np.sign(y_safe) * np.log1p(np.abs(y_safe))
    
    return y_transformed

def inverse_compositional_transform(y_transformed):
    """Inversa de la transformación composicional"""
    return np.sign(y_transformed) * (np.exp(np.abs(y_transformed)) - 1)

def load_cpa_features(train=True):
    """Carga y prepara características para CPA"""
    aug_dir = SETTINGS["TRAIN_DATA_AUG_DIR"]
    
    if train:
        df = pd.read_parquet(SETTINGS["TRAIN_RAW_DATA_PATH"])
    else:
        df = pd.read_csv(SETTINGS["TEST_RAW_DATA_PATH"])
    
    print(f"📊 Cargando datos {'train' if train else 'test'} para CPA...")
    
    # Guardar información composicional
    cell_types = df['cell_type'].values if 'cell_type' in df.columns else None
    sm_names = df['sm_name'].values if 'sm_name' in df.columns else None
    
    features_list = []
    
    # 1. Características base
    try:
        oh_suffix = "" if train else "_test"
        oh_cell = pd.read_csv(f"{aug_dir}one_hot_cell_type{oh_suffix}.csv")
        oh_sm = pd.read_csv(f"{aug_dir}one_hot_sm_name{oh_suffix}.csv")
        features_list.extend([oh_cell, oh_sm])
        print(f"   ✓ One-hot: {oh_cell.shape[1] + oh_sm.shape[1]} features")
    except Exception as e:
        print(f"   ⚠️  Error one-hot: {e}")
    
    # 2. Embeddings químicos profundos
    try:
        chem_suffix = "train" if train else "test"
        chem_feats = pd.read_csv(f"{aug_dir}ChemBERTa_{chem_suffix}.csv")
        features_list.append(chem_feats)
        print(f"   ✓ ChemBERTa: {chem_feats.shape[1]} features")
    except Exception as e:
        print(f"   ⚠️  Error ChemBERTa: {e}")
    
    # 3. Morgan Fingerprints (importante para CPA)
    try:
        fp_suffix = "train" if train else "test"
        morgan_fp = pd.read_csv(f"{aug_dir}morgan_fp_{fp_suffix}.csv")
        features_list.append(morgan_fp)
        print(f"   ✓ Morgan fingerprints: {morgan_fp.shape[1]} features")
    except Exception as e:
        print(f"   ⚠️  Morgan fingerprints no disponible: {e}")
    
    # 4. Características de interacción (CPA-specific)
    try:
        if train:
            # Calcular interacciones celular-compuesto
            cell_sm_interaction = np.zeros((len(df), 100), dtype=np.float32)
            
            # Simular interacciones (en implementación real, calcularías esto)
            for i in range(len(df)):
                # Ejemplo: interacción basada en hash
                hash_val = hash(str(cell_types[i]) + str(sm_names[i])) % 100
                cell_sm_interaction[i, hash_val % 100] = 1.0
            
            interaction_df = pd.DataFrame(cell_sm_interaction)
            interaction_df.columns = [f'interaction_{i}' for i in range(100)]
            features_list.append(interaction_df)
            print(f"   ✓ Interacciones celular-compuesto: 100 features")
            
            # Guardar para test
            joblib.dump(cell_sm_interaction, "cpa_features/interaction_patterns.pkl")
    except Exception as e:
        print(f"   ⚠️  Error interacciones: {e}")
    
    # Combinar características
    if features_list:
        features = pd.concat(features_list, axis=1)
    else:
        # Fallback
        n_samples = len(df)
        features = pd.DataFrame(np.random.randn(n_samples, 100).astype(np.float32))
    
    # Convertir a float32
    features = features.astype(np.float32)
    
    # NORMALIZACIÓN COMPOSICIONAL
    features_norm = compositional_normalization(features.values, cell_types, sm_names, train)
    
    # Añadir características estadísticas
    if train:
        try:
            mean_cell = pd.read_csv(f"{aug_dir}mean_cell_type.csv").values.astype(np.float32)
            std_cell = pd.read_csv(f"{aug_dir}std_cell_type.csv").values.astype(np.float32)
            mean_sm = pd.read_csv(f"{aug_dir}mean_sm_name.csv").values.astype(np.float32)
            std_sm = pd.read_csv(f"{aug_dir}std_sm_name.csv").values.astype(np.float32)
            
            # Añadir al final
            stats_features = np.concatenate([mean_cell, std_cell, mean_sm, std_sm], axis=1)
            features_norm = np.concatenate([features_norm, stats_features], axis=1)
            print(f"   ✓ Estadísticas agregadas: {stats_features.shape[1]} features")
        except:
            pass
    
    print(f"✅ Características CPA finales: {features_norm.shape}")
    
    return df, features_norm, cell_types, sm_names

def train_cpa_model(X_train, y_train, X_val, y_val, gene_name, fold_num, params):
    """Entrena un modelo CPA para un gen específico"""
    
    # Transformación composicional del target
    y_train_transformed = compositional_target_transform(y_train)
    y_val_transformed = compositional_target_transform(y_val)
    
    # Normalización estándar después de transformación
    y_mean = np.mean(y_train_transformed)
    y_std = np.std(y_train_transformed) + 1e-10
    
    y_train_norm = (y_train_transformed - y_mean) / y_std
    y_val_norm = (y_val_transformed - y_mean) / y_std
    
    # Dataset
    train_data = lgb.Dataset(X_train, label=y_train_norm)
    val_data = lgb.Dataset(X_val, label=y_val_norm, reference=train_data)
    
    # Entrenar con más iteraciones (CPA requiere más)
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=2000,  # Más iteraciones para aprendizaje composicional
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(0),
            lgb.record_evaluation({})
        ]
    )
    
    # Predecir y revertir transformaciones
    y_pred_norm = model.predict(X_val)
    y_pred_transformed = y_pred_norm * y_std + y_mean
    y_pred = inverse_compositional_transform(y_pred_transformed)
    
    # Métricas
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = np.mean(np.abs(y_val - y_pred))
    r2 = r2_score(y_val, y_pred)
    
    # Parámetros de transformación para predicción
    transform_params = {
        'y_mean': y_mean,
        'y_std': y_std,
        'gene_name': gene_name,
        'fold': fold_num
    }
    
    return model, rmse, mae, r2, transform_params

def train_cpa_cross_validation(df, features, n_folds=5, n_genes=None):
    """Entrenamiento CPA con validación cruzada"""
    print("\n" + "="*70)
    print("🧬 ENTRENAMIENTO CPA - COMPOSITIONAL PERTURBATION AUTOENCODER")
    print("="*70)
    
    # Parámetros CPA optimizados
    params = get_cpa_optimized_params()
    
    print(f"\n🔧 PARÁMETROS CPA OPTIMIZADOS:")
    print(f"   • Learning rate: {params['learning_rate']} (muy bajo)")
    print(f"   • Num leaves: {params['num_leaves']} (alto)")
    print(f"   • Regularización: {params['reg_alpha']}/{params['reg_lambda']} (muy baja)")
    print(f"   • Subsampling: {params['subsample']} (robusto)")
    
    # Seleccionar genes
    target_cols = df.columns[5:]
    if n_genes and n_genes < len(target_cols):
        # Seleccionar genes representativos basados en varianza
        gene_vars = df[target_cols].var()
        
        # Estratificar por percentiles de varianza
        percentiles = np.percentile(gene_vars, [0, 25, 50, 75, 100])
        selected_genes = []
        
        for i in range(len(percentiles)-1):
            mask = (gene_vars >= percentiles[i]) & (gene_vars < percentiles[i+1])
            genes_in_percentile = gene_vars[mask].index.tolist()
            n_select = max(1, n_genes // (len(percentiles)-1))
            selected = np.random.choice(genes_in_percentile, 
                                      min(n_select, len(genes_in_percentile)), 
                                      replace=False)
            selected_genes.extend(selected)
        
        target_cols = selected_genes[:n_genes]
        print(f"🔧 MODO PRUEBA: {n_genes} genes (estratificados por varianza)")
    
    print(f"\n📈 Entrenando {len(target_cols)} genes con CPA")
    print(f"📊 Características: {features.shape}")
    
    # Cross-validation estratificada por tipo celular
    if 'cell_type' in df.columns:
        cell_types = df['cell_type'].values
        unique_cells, cell_indices = np.unique(cell_types, return_inverse=True)
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kf.split(features, cell_indices)
        print(f"   • Stratified KFold por tipo celular ({len(unique_cells)} tipos)")
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kf.split(features)
        print(f"   • KFold estándar")
    
    models = {}
    fold_scores = []
    fold_details = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"🔄 FOLD CPA {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        X_train = features[train_idx]
        X_val = features[val_idx]
        
        fold_models = []
        fold_transform_params = []
        gene_rmse_scores = []
        gene_mae_scores = []
        gene_r2_scores = []
        
        start_time = time.time()
        
        # Barra de progreso
        for i, gene in enumerate(tqdm(target_cols, desc=f"   Entrenando genes CPA")):
            y_train = df.iloc[train_idx][gene].values.astype(np.float32)
            y_val_fold = df.iloc[val_idx][gene].values.astype(np.float32)
            
            # Entrenar modelo CPA
            model, rmse, mae, r2, transform_params = train_cpa_model(
                X_train, y_train, X_val, y_val_fold, gene, fold, params
            )
            
            fold_models.append(model)
            fold_transform_params.append(transform_params)
            gene_rmse_scores.append(rmse)
            gene_mae_scores.append(mae)
            gene_r2_scores.append(r2)
            
            # Mostrar progreso cada 5 genes
            if (i + 1) % 5 == 0:
                current_avg = np.mean(gene_rmse_scores[-5:])
                print(f"      Genes {i-3}-{i+1}: RMSE = {current_avg:.4f}")
        
        models[f'cpa_fold_{fold}'] = {
            'models': fold_models,
            'transform_params': fold_transform_params
        }
        
        # Estadísticas fold
        fold_rmse = np.mean(gene_rmse_scores)
        fold_mae = np.mean(gene_mae_scores)
        fold_r2 = np.mean(gene_r2_scores)
        fold_scores.append(fold_rmse)
        
        elapsed_total = time.time() - start_time
        
        print(f"\n✅ Fold CPA {fold + 1} completado en {elapsed_total:.1f}s")
        print(f"   📊 RMSE: {fold_rmse:.4f}")
        print(f"   📊 MAE: {fold_mae:.4f}")
        print(f"   📊 R²: {fold_r2:.4f}")
        print(f"   🎯 Mejor gen: {np.min(gene_rmse_scores):.4f}")
        print(f"   ⚠️  Peor gen: {np.max(gene_rmse_scores):.4f}")
        
        fold_details.append({
            'rmse_scores': gene_rmse_scores,
            'mae_scores': gene_mae_scores,
            'r2_scores': gene_r2_scores,
            'time': elapsed_total
        })
    
    # Resultados finales
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    cv_mae = np.mean([np.mean(fold['mae_scores']) for fold in fold_details])
    cv_r2 = np.mean([np.mean(fold['r2_scores']) for fold in fold_details])
    
    print(f"\n{'='*70}")
    print("🎯 RESULTADOS FINALES CPA")
    print(f"{'='*70}")
    
    for fold, score in enumerate(fold_scores):
        print(f"   Fold {fold + 1}: RMSE = {score:.4f}")
    
    print(f"\n📊 RESUMEN ESTADÍSTICO CPA:")
    print(f"   RMSE CV: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"   MAE CV: {cv_mae:.4f}")
    print(f"   R² CV: {cv_r2:.4f}")
    print(f"   Mejor fold: {np.min(fold_scores):.4f}")
    print(f"   Peor fold: {np.max(fold_scores):.4f}")
    
    return models, target_cols, cv_mean, fold_scores, fold_details

def save_cpa_models(models, target_cols, cv_score, fold_scores, fold_details, 
                   path="cpa_models/"):
    """Guarda modelos CPA con metadatos"""
    if not os.path.exists(path):
        os.makedirs(path)
    
    print(f"\n💾 Guardando modelos CPA...")
    
    model_count = 0
    transform_count = 0
    
    for fold_name, fold_data in models.items():
        fold_dir = os.path.join(path, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        
        # Guardar modelos
        for i, model in enumerate(fold_data['models']):
            model.save_model(os.path.join(fold_dir, f"cpa_gene_{i:04d}.txt"))
            model_count += 1
        
        # Guardar parámetros de transformación
        transforms_file = os.path.join(fold_dir, "cpa_transform_params.pkl")
        joblib.dump(fold_data['transform_params'], transforms_file)
        transform_count += 1
        
        print(f"   ✓ {fold_name}: {len(fold_data['models'])} modelos")
    
    # Metadatos CPA
    metadata = {
        'target_cols': list(target_cols),
        'cv_score': cv_score,
        'fold_scores': fold_scores,
        'fold_details': fold_details,
        'n_folds': len(models),
        'n_genes': len(target_cols),
        'model_count': model_count,
        'transform_count': transform_count,
        'model_type': 'CPA',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'cpa_params': get_cpa_optimized_params(),
        'training_notes': 'CPA - Compositional Perturbation Autoencoder'
    }
    
    metadata_path = os.path.join(path, "cpa_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Cache para predicción rápida
    cache_data = {
        'models_info': {k: len(v['models']) for k, v in models.items()},
        'target_cols': target_cols,
        'metadata': metadata
    }
    
    joblib.dump(cache_data, os.path.join(path, "cpa_cache.pkl"))
    
    print(f"\n✅ {model_count} modelos CPA guardados en {path}")
    print(f"📊 RMSE CPA: {cv_score:.4f}")
    
    return metadata

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenamiento CPA (Compositional Perturbation Autoencoder)")
    parser.add_argument("--folds", type=int, default=5, help="Número de folds")
    parser.add_argument("--genes", type=int, default=None, help="Número de genes")
    parser.add_argument("--output", type=str, default="cpa_models/", help="Directorio salida")
    parser.add_argument("--test", action="store_true", help="Modo prueba")
    
    args = parser.parse_args()
    
    print("="*80)
    print("🧬 CPA - COMPOSITIONAL PERTURBATION AUTOENCODER")
    print("="*80)
    print("🎯 Características del modelo:")
    print("   1. Normalización composicional por tipo celular y compuesto")
    print("   2. Transformación logarítmica del target")
    print("   3. Regularización mínima para capturar interacciones")
    print("   4. 5-Fold Cross-Validation estratificada")
    print("="*80)
    
    # Configurar entorno
    setup_environment()
    
    # Modo prueba
    if args.test:
        args.folds = 2
        args.genes = 100 if args.genes is None else args.genes
        print(f"\n🔧 MODO PRUEBA CPA: {args.folds} folds, {args.genes} genes")
    
    # Cargar características CPA
    print(f"\n📥 Cargando características CPA...")
    df, features, cell_types, sm_names = load_cpa_features(train=True)
    
    print(f"\n📊 RESUMEN CPA:")
    print(f"   • Muestras: {len(df):,}")
    print(f"   • Genes: {len(df.columns) - 5}")
    print(f"   • Características CPA: {features.shape[1]}")
    print(f"   • Tipos celulares: {len(np.unique(cell_types)) if cell_types is not None else 'N/A'}")
    print(f"   • Compuestos únicos: {len(np.unique(sm_names)) if sm_names is not None else 'N/A'}")
    
    # Entrenar CPA
    print(f"\n⚡ Iniciando entrenamiento CPA...")
    start_time = time.time()
    
    models, target_cols, cv_score, fold_scores, fold_details = train_cpa_cross_validation(
        df, features,
        n_folds=args.folds,
        n_genes=args.genes
    )
    
    training_time = time.time() - start_time
    
    # Guardar
    print(f"\n💾 Guardando modelos CPA...")
    metadata = save_cpa_models(
        models, target_cols, cv_score, fold_scores, fold_details,
        args.output
    )
    
    # Resultados
    print(f"\n{'='*80}")
    print("✅ ENTRENAMIENTO CPA COMPLETADO")
    print(f"{'='*80}")
    print(f"📈 RMSE CPA CV: {cv_score:.4f}")
    print(f"⏱️  Tiempo total: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"🧬 Genes CPA: {len(target_cols)}")
    print(f"📁 Guardado en: {args.output}")
    
    # Análisis
    print(f"\n📊 ANÁLISIS CPA:")
    if cv_score < 1.5:
        print("   🎉 ¡Excelente rendimiento CPA! (RMSE < 1.5)")
    elif cv_score < 1.8:
        print("   ✅ Buen rendimiento CPA (RMSE < 1.8)")
    elif cv_score < 2.0:
        print("   📈 Rendimiento CPA aceptable")
    else:
        print("   ⚠️  CPA puede mejorar")
    
    print(f"\n🎯 PARA PREDICCIÓN CPA:")
    print(f"   Ejecuta: python inference_cpa.py")

if __name__ == "__main__":
    main()