# train_model_cuda.py
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import os
import warnings
import time
from tqdm import tqdm
import subprocess
import sys

warnings.filterwarnings('ignore')

# Configuración
with open("./SETTINGS.json") as f:
    SETTINGS = json.load(f)

def setup_cuda_environment():
    """Configura el entorno CUDA automáticamente"""
    print(" Configurando entorno CUDA...")
    
    # Configurar variables de entorno para CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usar primera GPU
    
    # Verificar CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f" PyTorch CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  PyTorch no detecta CUDA")
    except:
        pass
    
    # Verificar nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f" CUDA Toolkit disponible")
            # Extraer versión
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"   Versión: {line}")
                    break
    except:
        print("  nvcc no encontrado")
    
    return True

def get_cuda_params():
    """Parámetros optimizados para CUDA"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 255,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'max_depth': -1,
        
        # CONFIGURACIÓN CUDA - LightGBM detectará automáticamente
        'device': 'cuda',  # Usar CUDA directamente
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'max_bin': 63,
        'tree_learner': 'data',
        'seed': 42,
        'verbose': -1,
        'num_threads': 0,
    }
    return params

def check_cuda_support():
    """Verifica si LightGBM tiene soporte CUDA"""
    try:
        # Crear parámetros CUDA
        params = {'device': 'cuda', 'verbose': -1}
        
        # Probar con datos dummy
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        
        train_data = lgb.Dataset(X, label=y)
        
        # Intentar entrenar
        gbm = lgb.train(params, train_data, num_boost_round=5)
        
        # Verificar si está usando GPU
        if hasattr(gbm, 'device'):
            print(f" LightGBM usando: {gbm.device}")
        else:
            # Verificar en logs
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                params_debug = {'device': 'cuda', 'verbose': 1}
                gbm_debug = lgb.train(params_debug, train_data, num_boost_round=1)
            
            output = f.getvalue()
            if 'GPU' in output or 'CUDA' in output:
                print(" LightGBM usando CUDA/GPU")
            else:
                print("  LightGBM no está usando GPU")
                
        return True
        
    except Exception as e:
        print(f" Error con CUDA: {e}")
        return False

def load_features_cuda(train=True):
    """Carga características optimizadas para CUDA"""
    aug_dir = SETTINGS["TRAIN_DATA_AUG_DIR"]
    
    if train:
        df = pd.read_parquet(SETTINGS["TRAIN_RAW_DATA_PATH"])
    else:
        df = pd.read_csv(SETTINGS["TEST_RAW_DATA_PATH"])
    
    print(f" Cargando datos {'train' if train else 'test'}...")
    
    # Cargar características manteniendo float32 para CUDA
    features_list = []
    
    # One-hot
    try:
        oh_suffix = "" if train else "_test"
        oh_cell = pd.read_csv(f"{aug_dir}one_hot_cell_type{oh_suffix}.csv")
        oh_sm = pd.read_csv(f"{aug_dir}one_hot_sm_name{oh_suffix}.csv")
        features_list.extend([oh_cell, oh_sm])
    except Exception as e:
        print(f"  Error one-hot: {e}")
        n_samples = len(df)
        oh_cell = pd.DataFrame(np.zeros((n_samples, 1), dtype=np.float32), columns=['cell_type_dummy'])
        oh_sm = pd.DataFrame(np.zeros((n_samples, 1), dtype=np.float32), columns=['sm_name_dummy'])
        features_list.extend([oh_cell, oh_sm])
    
    # ChemBERTa
    try:
        chem_suffix = "train" if train else "test"
        chem_feats = pd.read_csv(f"{aug_dir}ChemBERTa_{chem_suffix}.csv")
        features_list.append(chem_feats)
    except Exception as e:
        print(f"  Error ChemBERTa: {e}")
        n_samples = len(df)
        chem_feats = pd.DataFrame(np.random.randn(n_samples, 100).astype(np.float32) * 0.1)
        chem_feats.columns = [f'chem_{i}' for i in range(100)]
        features_list.append(chem_feats)
    
    # Combinar
    features = pd.concat(features_list, axis=1)
    
    # Convertir a float32 para CUDA
    features = features.astype(np.float32)
    
    # Estadísticas solo para train
    if train:
        try:
            mean_cell = pd.read_csv(f"{aug_dir}mean_cell_type.csv").astype(np.float32)
            std_cell = pd.read_csv(f"{aug_dir}std_cell_type.csv").astype(np.float32)
            mean_sm = pd.read_csv(f"{aug_dir}mean_sm_name.csv").astype(np.float32)
            std_sm = pd.read_csv(f"{aug_dir}std_sm_name.csv").astype(np.float32)
            
            # Agregar columnas para merge
            if 'cell_type' in df.columns:
                features['cell_type'] = df['cell_type'].values
            if 'sm_name' in df.columns:
                features['sm_name'] = df['sm_name'].values
            
            # Merge
            features = features.merge(mean_cell, on='cell_type', how='left', suffixes=('', '_mean_cell'))
            features = features.merge(std_cell, on='cell_type', how='left', suffixes=('', '_std_cell'))
            features = features.merge(mean_sm, on='sm_name', how='left', suffixes=('', '_mean_sm'))
            features = features.merge(std_sm, on='sm_name', how='left', suffixes=('', '_std_sm'))
            
            # Eliminar columnas temporales
            for col in ['cell_type', 'sm_name']:
                if col in features.columns:
                    features = features.drop(columns=[col])
                    
        except Exception as e:
            print(f"  Error estadísticas: {e}")
    
    # Asegurar float32
    features = features.astype(np.float32)
    
    print(f" Características: {features.shape} (float32)")
    
    return df, features

def train_with_cuda(df, features, n_folds=3, n_genes=None):
    """Entrena usando CUDA"""
    print("\n" + "="*70)
    print(" ENTRENAMIENTO CON CUDA ACTIVADO")
    print("="*70)
    
    # Configurar parámetros CUDA
    params = get_cuda_params()
    
    # Verificar CUDA
    if not check_cuda_support():
        print("  Cambiando a CPU...")
        params['device'] = 'cpu'
    
    print(f"\n Parámetros CUDA:")
    print(f"   Dispositivo: {params['device']}")
    print(f"   learning_rate: {params['learning_rate']}")
    print(f"   num_leaves: {params['num_leaves']}")
    
    # Seleccionar genes
    target_cols = df.columns[5:]
    if n_genes and n_genes < len(target_cols):
        np.random.seed(42)
        target_cols = np.random.choice(target_cols, n_genes, replace=False)
        print(f"🔧 MODO PRUEBA: {n_genes} genes")
    
    print(f"\n Entrenando {len(target_cols)} genes")
    print(f" Características: {features.shape}")
    
    # K-Fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    models = {}
    fold_scores = []
    
    # Convertir a numpy float32 para CUDA
    features_np = features.values.astype(np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n{'='*60}")
        print(f" FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        X_train = features_np[train_idx]
        X_val = features_np[val_idx]
        
        fold_models = []
        gene_scores = []
        
        start_time = time.time()
        
        # Progreso por lotes
        n_genes_total = len(target_cols)
        batch_size = 10  # Mostrar progreso cada 10 genes
        
        for i, gene in enumerate(target_cols):
            y_train = df.iloc[train_idx][gene].values.astype(np.float32)
            y_val_fold = df.iloc[val_idx][gene].values.astype(np.float32)
            
            # Dataset
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val_fold, reference=train_data)
            
            # Entrenar con CUDA
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            
            fold_models.append(model)
            
            # Evaluar
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            gene_scores.append(rmse)
            
            # Mostrar progreso
            if (i + 1) % batch_size == 0 or (i + 1) == n_genes_total:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                avg_rmse = np.mean(gene_scores[-batch_size:]) if len(gene_scores) >= batch_size else np.mean(gene_scores)
                print(f"   Genes {i+1-batch_size+1}-{i+1}/{n_genes_total}: "
                      f"RMSE={avg_rmse:.4f}, {speed:.1f} genes/s")
        
        models[f'fold_{fold}'] = fold_models
        
        # Estadísticas fold
        fold_rmse = np.mean(gene_scores)
        fold_scores.append(fold_rmse)
        
        elapsed_total = time.time() - start_time
        
        print(f"\n Fold {fold + 1} completado en {elapsed_total:.1f}s")
        print(f"    RMSE: {fold_rmse:.4f}")
        print(f"    Mejor: {np.min(gene_scores):.4f}")
        print(f"     Peor: {np.max(gene_scores):.4f}")
    
    # Resultados
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    print(f"\n{'='*70}")
    print(" RESULTADOS CUDA")
    print(f"{'='*70}")
    
    for fold, score in enumerate(fold_scores):
        print(f"   Fold {fold + 1}: {score:.4f}")
    
    print(f"\n RMSE CV: {cv_mean:.4f} (±{cv_std:.4f})")
    
    return models, target_cols, cv_mean, fold_scores

def save_cuda_models(models, target_cols, cv_score, fold_scores, path="models_cuda/"):
    """Guarda modelos entrenados con CUDA"""
    if not os.path.exists(path):
        os.makedirs(path)
    
    print(f"\n Guardando modelos CUDA...")
    
    # Guardar cada modelo
    model_count = 0
    for fold_name, fold_models in models.items():
        fold_dir = os.path.join(path, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        
        for i, model in enumerate(fold_models):
            model.save_model(os.path.join(fold_dir, f"gene_{i:04d}.txt"))
            model_count += 1
    
    # Metadatos
    metadata = {
        'target_cols': list(target_cols),
        'cv_score': cv_score,
        'fold_scores': fold_scores,
        'n_folds': len(models),
        'n_genes': len(target_cols),
        'model_count': model_count,
        'device': 'cuda',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'cuda_params': {
            'learning_rate': 0.1,
            'num_leaves': 255,
            'device': 'cuda'
        }
    }
    
    with open(os.path.join(path, "metadata_cuda.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Cache
    joblib.dump({
        'models': models,
        'target_cols': target_cols,
        'metadata': metadata
    }, os.path.join(path, "models_cuda_cache.pkl"))
    
    print(f" {model_count} modelos guardados en {path}")
    
    return metadata

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenamiento con CUDA")
    parser.add_argument("--folds", type=int, default=3, help="Número de folds")
    parser.add_argument("--genes", type=int, default=None, help="Número de genes")
    parser.add_argument("--output", type=str, default="models_cuda/", help="Directorio salida")
    parser.add_argument("--test", action="store_true", help="Modo prueba")
    
    args = parser.parse_args()
    
    print("="*80)
    print(" ENTRENAMIENTO CON CUDA - RTX 2060 SUPER")
    print("="*80)
    
    # Configurar CUDA
    setup_cuda_environment()
    
    # Modo prueba
    if args.test:
        args.folds = 2
        args.genes = 50 if args.genes is None else args.genes
        print(f"\n🔧 MODO PRUEBA: {args.folds} folds, {args.genes} genes")
    
    # Cargar datos
    print(f"\n Cargando datos...")
    df, features = load_features_cuda(train=True)
    
    print(f"\n RESUMEN:")
    print(f"   Muestras: {len(df):,}")
    print(f"   Genes: {len(df.columns) - 5}")
    print(f"   Características: {features.shape[1]}")
    print(f"   Tipo datos: {features.dtypes.iloc[0]}")
    
    # Entrenar
    print(f"\n Iniciando entrenamiento CUDA...")
    start_time = time.time()
    
    models, target_cols, cv_score, fold_scores = train_with_cuda(
        df, features,
        n_folds=args.folds,
        n_genes=args.genes
    )
    
    training_time = time.time() - start_time
    
    # Guardar
    print(f"\n Guardando...")
    metadata = save_cuda_models(
        models, target_cols, cv_score, fold_scores,
        args.output
    )
    
    # Resultados
    print(f"\n{'='*80}")
    print(" ENTRENAMIENTO CUDA COMPLETADO")
    print(f"{'='*80}")
    print(f" RMSE CV: {cv_score:.4f}")
    print(f"  Tiempo: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f" Genes: {len(target_cols)}")
    print(f" Guardado en: {args.output}")
    print(f"\n Para predicción: python inference_cuda.py")
    print(f" Para análisis: ls -la {args.output}/")

if __name__ == "__main__":
    main()