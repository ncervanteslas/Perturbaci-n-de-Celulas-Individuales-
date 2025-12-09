# prepare_data_debug.py
from argparse import Namespace
import os
import json
import pandas as pd
import numpy as np
import torch
import random
from transformers import AutoTokenizer, AutoModel

def seed_everything(seed=42):
    """Fija semilla para reproducibilidad"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def one_hot_encode(train_cat, test_cat, out_dir):
    """
    Crea codificación one-hot para variables categóricas
    """
    print("  Creando one-hot encodings...")
    
    # Combinar train y test para tener todas las categorías
    combined = pd.concat([train_cat, test_cat], axis=0)
    
    # One-hot para cell_type
    cell_type_dummies = pd.get_dummies(combined['cell_type'], prefix='cell_type')
    cell_type_dummies.iloc[:len(train_cat)].to_csv(f'{out_dir}one_hot_cell_type.csv', index=False)
    cell_type_dummies.iloc[len(train_cat):].reset_index(drop=True).to_csv(
        f'{out_dir}one_hot_cell_type_test.csv', index=False
    )
    
    # One-hot para sm_name
    sm_name_dummies = pd.get_dummies(combined['sm_name'], prefix='sm_name')
    sm_name_dummies.iloc[:len(train_cat)].to_csv(f'{out_dir}one_hot_sm_name.csv', index=False)
    sm_name_dummies.iloc[len(train_cat):].reset_index(drop=True).to_csv(
        f'{out_dir}one_hot_sm_name_test.csv', index=False
    )
    
    print(f"  One-hot guardados en {out_dir}")

def save_ChemBERTa_features_debug(smiles_list, out_dir, on_train_data=True):
    """
    Extrae características de ChemBERTa para moléculas SMILES (con debugging)
    """
    print(f"\n{'='*50}")
    print(f"EXTRACCIÓN DE CARACTERÍSTICAS ChemBERTa")
    print(f"{'='*50}")
    print(f"  Modo: {'Train' if on_train_data else 'Test'}")
    print(f"  Número de moléculas: {len(smiles_list)}")
    print(f"  SMILES no nulos: {sum(1 for s in smiles_list if s is not None)}")
    
    # Ver primeros SMILES
    print(f"  Primeros SMILES: {smiles_list[:3] if len(smiles_list) > 3 else smiles_list}")
    
    try:
        # Cargar modelo ChemBERTa
        print("\n  1. Descargando tokenizer de ChemBERTa...")
        tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        print("     ✓ Tokenizer cargado")
        
        print("  2. Descargando modelo ChemBERTa...")
        model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        print("     ✓ Modelo cargado")
        
        # Configurar dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  3. Usando dispositivo: {device}")
        model.to(device)
        model.eval()
        
        # Procesar en lotes
        batch_size = 16  # Reducido para debugging
        all_features = []
        valid_smiles = []
        
        print(f"  4. Procesando en lotes de {batch_size}...")
        
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            
            # Filtrar None values y convertir a string
            clean_batch = []
            for j, smile in enumerate(batch_smiles):
                if pd.isna(smile) or smile is None:
                    print(f"    ⚠️ SMILES nulo en posición {i+j}")
                    # Usar SMILES dummy para molécula desconocida
                    clean_batch.append("C")
                else:
                    clean_batch.append(str(smile))
            
            if not clean_batch:
                print(f"    Lote {i//batch_size + 1}: Vacío (todos nulos)")
                continue
            
            batch_num = i//batch_size + 1
            total_batches = (len(smiles_list) - 1)//batch_size + 1
            print(f"    Lote {batch_num}/{total_batches}: {len(clean_batch)} moléculas válidas")
            
            try:
                # Tokenizar
                inputs = tokenizer(
                    clean_batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(device)
                
                print(f"      Longitud del input: {inputs['input_ids'].shape}")
                
                # Obtener embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Usar el embedding del token [CLS] (posición 0)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    all_features.append(cls_embeddings)
                    valid_smiles.extend(clean_batch)
                
                print(f"      Embeddings generados: {cls_embeddings.shape}")
                
            except Exception as e:
                print(f"      ERROR en tokenización: {e}")
                print(f"      SMILES problemáticos: {clean_batch}")
                # Crear embeddings dummy para este lote
                dummy_embeddings = np.zeros((len(clean_batch), 768))
                all_features.append(dummy_embeddings)
                valid_smiles.extend(clean_batch)
        
        if len(all_features) == 0:
            print("\n  ⚠️ ADVERTENCIA: No se generaron características!")
            # Crear características dummy como fallback
            n_samples = len([s for s in smiles_list if s is not None])
            features_array = np.zeros((n_samples, 768))
            print(f"  Creando características dummy: {features_array.shape}")
        else:
            # Concatenar todos los embeddings
            features_array = np.vstack(all_features)
            print(f"\n  ✓ Embeddings concatenados: {features_array.shape}")
        
        # Convertir a DataFrame
        features_df = pd.DataFrame(features_array)
        features_df.columns = [f"chemberta_{i}" for i in range(features_array.shape[1])]
        
        # Guardar
        suffix = "train" if on_train_data else "test"
        output_path = f'{out_dir}ChemBERTa_{suffix}.csv'
        features_df.to_csv(output_path, index=False)
        
        print(f"\n  ✓ Características ChemBERTa guardadas en: {output_path}")
        print(f"  Dimensiones finales: {features_df.shape}")
        
        # Guardar también un resumen
        summary_path = f'{out_dir}ChemBERTa_{suffix}_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"SMILES procesados: {len(valid_smiles)}\n")
            f.write(f"Características generadas: {features_df.shape}\n")
            f.write(f"Rango de valores: [{features_array.min():.4f}, {features_array.max():.4f}]\n")
            f.write(f"Media: {features_array.mean():.4f}\n")
            f.write(f"Desviación estándar: {features_array.std():.4f}\n")
        
        print(f"  Resumen guardado en: {summary_path}")
        
        return features_df
        
    except Exception as e:
        print(f"\n  ✗ ERROR CRÍTICO en save_ChemBERTa_features: {e}")
        print(f"  Tipo de error: {type(e).__name__}")
        
        import traceback
        traceback.print_exc()
        
        print("\n  Creando características dummy como fallback...")
        
        # Crear características dummy
        n_samples = len([s for s in smiles_list if s is not None])
        if n_samples == 0:
            n_samples = len(smiles_list)
        
        features_array = np.zeros((n_samples, 768))
        print(f"  Dimensiones dummy: {features_array.shape}")
        
        features_df = pd.DataFrame(features_array)
        features_df.columns = [f"chemberta_{i}" for i in range(features_array.shape[1])]
        
        suffix = "train" if on_train_data else "test"
        output_path = f'{out_dir}ChemBERTa_{suffix}.csv'
        features_df.to_csv(output_path, index=False)
        
        print(f"  ✓ Características dummy guardadas en: {output_path}")
        
        return features_df

if __name__ == "__main__":
    ## Seed for reproducibility
    seed_everything()
    
    with open("./SETTINGS.json") as file:
        settings = json.load(file)
    
    ## Read data
    print("\n" + "="*60)
    print("PREPARANDO DATOS - MODO DEBUG")
    print("="*60)
    
    print("\n1. Leyendo datos...")
    de_train = pd.read_parquet(settings["TRAIN_RAW_DATA_PATH"])
    id_map = pd.read_csv(settings["TEST_RAW_DATA_PATH"])
    
    print(f"   Train data: {de_train.shape}")
    print(f"   Test data: {id_map.shape}")
    print(f"   Columnas train: {list(de_train.columns[:10])}...")
    
    ## Create data augmentation
    print("\n2. Creando características aumentadas...")
    de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]
    de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]
    
    mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
    mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()
    std_cell_type = de_cell_type.groupby('cell_type').std().reset_index()
    std_sm_name = de_sm_name.groupby('sm_name').std().reset_index()
    
    # Quantiles
    cell_types = de_cell_type.groupby('cell_type').quantile(0.1).reset_index()['cell_type']
    quantiles_cell_type = pd.concat(
        [pd.DataFrame(cell_types)] + 
        [de_cell_type.groupby('cell_type')[col].quantile([0.25, 0.50, 0.75], interpolation='linear')
         .unstack().reset_index(drop=True) for col in list(de_train.columns)[5:10]],  # Solo primeras 5 columnas para debug
        axis=1
    )
    
    ## Save data augmentation features
    print("\n3. Guardando características aumentadas...")
    if not os.path.exists(settings["TRAIN_DATA_AUG_DIR"]):
        os.mkdir(settings["TRAIN_DATA_AUG_DIR"])
        print(f"   Directorio creado: {settings['TRAIN_DATA_AUG_DIR']}")
    
    mean_cell_type.to_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_cell_type.csv', index=False)
    std_cell_type.to_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_cell_type.csv', index=False)
    mean_sm_name.to_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}mean_sm_name.csv', index=False)
    std_sm_name.to_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}std_sm_name.csv', index=False)
    quantiles_cell_type.to_csv(f'{settings["TRAIN_DATA_AUG_DIR"]}quantiles_cell_type.csv', index=False)
    
    print(f"   Archivos guardados en {settings['TRAIN_DATA_AUG_DIR']}")
    
    ## Create one hot encoding features
    print("\n4. Creando one-hot encodings...")
    one_hot_encode(de_train[["cell_type", "sm_name"]], id_map[["cell_type", "sm_name"]], 
                   out_dir=settings["TRAIN_DATA_AUG_DIR"])
    
    ## Prepare ChemBERTa features - TRAIN
    print("\n5. Extrayendo características ChemBERTa (TRAIN)...")
    train_smiles = de_train["SMILES"].tolist()
    save_ChemBERTa_features_debug(train_smiles, 
                                 out_dir=settings["TRAIN_DATA_AUG_DIR"], 
                                 on_train_data=True)
    
    ## Prepare ChemBERTa features - TEST
    print("\n6. Extrayendo características ChemBERTa (TEST)...")
    # Mapeo de sm_name a SMILES para test
    sm_name2smiles = {smname: smiles for smname, smiles in zip(de_train['sm_name'], de_train['SMILES'])}
    test_smiles = []
    
    for sm_name in id_map['sm_name'].values:
        if sm_name in sm_name2smiles:
            test_smiles.append(sm_name2smiles[sm_name])
        else:
            print(f"   ⚠️ sm_name no encontrado: {sm_name}")
            test_smiles.append(None)
    
    print(f"   SMILES de test encontrados: {sum(1 for s in test_smiles if s is not None)}/{len(test_smiles)}")
    
    save_ChemBERTa_features_debug(test_smiles, 
                                 out_dir=settings["TRAIN_DATA_AUG_DIR"], 
                                 on_train_data=False)
    
    print("\n" + "="*60)
    print("¡PROCESO COMPLETADO!")
    print("="*60)
    
    # Listar archivos generados
    print("\nArchivos generados en features/:")
    features_dir = settings["TRAIN_DATA_AUG_DIR"]
    if os.path.exists(features_dir):
        for file in os.listdir(features_dir):
            file_path = os.path.join(features_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"  - {file} ({size:.1f} KB)")