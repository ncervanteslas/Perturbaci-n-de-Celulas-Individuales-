# inference_cpa.py
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
from tqdm import tqdm
import warnings
import time
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

# Configuración
SETTINGS = {
    "TRAIN_RAW_DATA_PATH": "./data/de_train.parquet",
    "TEST_RAW_DATA_PATH": "./data/id_map.csv",
    "TRAIN_DATA_AUG_DIR": "./features/"
}

def compositional_normalization_inference(features, cell_types, sm_names):
    """Normalización composicional para inferencia"""
    try:
        # Cargar scalers
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
        print("⚠️  Usando escalado global")
        scaler = RobustScaler()
        features_norm = scaler.fit_transform(features)
    
    return features_norm.astype(np.float32)

def load_cpa_test_features():
    """Carga características de test para CPA"""
    print("📥 Cargando características CPA test...")
    
    # ID map
    id_map = pd.read_csv(SETTINGS["TEST_RAW_DATA_PATH"])
    cell_types = id_map['cell_type'].values if 'cell_type' in id_map.columns else None
    sm_names = id_map['sm_name'].values if 'sm_name' in id_map.columns else None
    
    # Cargar características base
    aug_dir = SETTINGS["TRAIN_DATA_AUG_DIR"]
    features_list = []
    
    # One-hot
    try:
        oh_cell = pd.read_csv(f"{aug_dir}one_hot_cell_type_test.csv")
        oh_sm = pd.read_csv(f"{aug_dir}one_hot_sm_name_test.csv")
        features_list.extend([oh_cell, oh_sm])
    except:
        n_samples = len(id_map)
        dummy = pd.DataFrame(np.zeros((n_samples, 1), dtype=np.float32))
        features_list.extend([dummy, dummy])
    
    # ChemBERTa
    try:
        chem_feats = pd.read_csv(f"{aug_dir}ChemBERTa_test.csv")
        features_list.append(chem_feats)
    except:
        n_samples = len(id_map)
        dummy = pd.DataFrame(np.random.randn(n_samples, 100).astype(np.float32))
        features_list.append(dummy)
    
    # Combinar
    features = pd.concat(features_list, axis=1).astype(np.float32)
    
    # Aplicar normalización composicional
    features_norm = compositional_normalization_inference(features.values, cell_types, sm_names)
    
    print(f"✅ Características CPA: {features_norm.shape}")
    
    return id_map, features_norm

def load_cpa_models():
    """Carga modelos CPA"""
    print("\n🔄 Cargando modelos CPA...")
    
    models_dir = "cpa_models"
    if not os.path.exists(models_dir):
        print(f"❌ No se encuentra {models_dir}")
        return None, None, None, None
    
    # Cargar metadata
    try:
        with open(f"{models_dir}/cpa_metadata.json", "r") as f:
            metadata = json.load(f)
        
        gene_names = metadata['target_cols']
        print(f"✅ CPA metadata cargada: {len(gene_names)} genes")
        
    except:
        print("⚠️  No se pudo cargar metadata CPA")
        return None, None, None, None
    
    # Cargar modelos por fold
    models_by_fold = {}
    transforms_by_fold = {}
    
    fold_dirs = [d for d in os.listdir(models_dir) if d.startswith('cpa_fold_')]
    
    for fold_dir in fold_dirs:
        fold_path = os.path.join(models_dir, fold_dir)
        
        # Cargar modelos
        model_files = [f for f in os.listdir(fold_path) 
                      if f.endswith('.txt') and f.startswith('cpa_gene_')]
        model_files = sorted(model_files)
        
        fold_models = []
        for model_file in tqdm(model_files, desc=f"Cargando {fold_dir}"):
            model = lgb.Booster(model_file=os.path.join(fold_path, model_file))
            fold_models.append(model)
        
        # Cargar transformaciones
        transforms_path = os.path.join(fold_path, "cpa_transform_params.pkl")
        if os.path.exists(transforms_path):
            fold_transforms = joblib.load(transforms_path)
        else:
            fold_transforms = []
        
        models_by_fold[fold_dir] = fold_models
        transforms_by_fold[fold_dir] = fold_transforms
    
    return models_by_fold, transforms_by_fold, gene_names, metadata

def inverse_compositional_transform(y_transformed):
    """Inversa de la transformación CPA"""
    return np.sign(y_transformed) * (np.exp(np.abs(y_transformed)) - 1)

def predict_cpa(models_by_fold, transforms_by_fold, features):
    """Predicción CPA"""
    if not models_by_fold:
        return None
    
    first_fold = list(models_by_fold.keys())[0]
    n_samples = features.shape[0]
    n_genes = len(models_by_fold[first_fold])
    n_folds = len(models_by_fold)
    
    print(f"\n🔮 Predicción CPA...")
    print(f"   Muestras: {n_samples}")
    print(f"   Genes: {n_genes}")
    print(f"   Folds: {n_folds}")
    
    predictions = np.zeros((n_samples, n_genes), dtype=np.float32)
    
    for fold_name, fold_models in models_by_fold.items():
        print(f"\n   📊 Procesando {fold_name}...")
        
        fold_transforms = transforms_by_fold.get(fold_name, [])
        
        for gene_idx in tqdm(range(n_genes), desc="     Genes"):
            model = fold_models[gene_idx]
            
            # Predecir (valores normalizados)
            pred_norm = model.predict(features)
            
            # Aplicar transformación inversa si hay parámetros
            if gene_idx < len(fold_transforms):
                transform = fold_transforms[gene_idx]
                if transform:
                    y_mean = transform.get('y_mean', 0)
                    y_std = transform.get('y_std', 1)
                    pred_transformed = pred_norm * y_std + y_mean
                    pred = inverse_compositional_transform(pred_transformed)
                else:
                    pred = pred_norm
            else:
                pred = pred_norm
            
            # Acumular para promedio
            predictions[:, gene_idx] += pred / n_folds
    
    return predictions

def create_cpa_submission(predictions, id_map, gene_names):
    """Crea submission para CPA"""
    print(f"\n💾 Creando submission CPA...")
    
    # DataFrame ancho
    submission_wide = pd.DataFrame(predictions, columns=gene_names[:predictions.shape[1]])
    submission_wide.insert(0, 'id', id_map['id'].values)
    
    # Convertir a largo
    submission_long = submission_wide.melt(
        id_vars=['id'],
        var_name='gene',
        value_name='target'
    )
    
    # Guard