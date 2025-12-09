# Perturbación de Células Individuales - Pipeline de Predicción Génica

Pipeline completo para predecir expresión génica en respuesta a perturbaciones químicas, implementando modelos avanzados de machine learning incluyendo CPA (Compositional Perturbation Autoencoder) y embeddings de moléculas con ChemBERTa.

##  Características

- **CPA (Compositional Perturbation Autoencoder)**: Modelo especializado que captura interacciones complejas
- **ChemBERTa**: Representaciones de moléculas usando transformers pre-entrenados
- **LightGBM con CUDA**: Entrenamiento acelerado por GPU
- **Pipeline completo**: Desde preprocesamiento hasta predicción
- **Validación cruzada**: 5-fold estratificada

## Comenzar

1. Clonar repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Colocar datos en `data/`
4. Ejecutar: `python scripts/prepare_data.py`
5. Entrenar: `python scripts/train_cpa.py`
6. Predecir: `python scripts/predict_cpa.py`

## Resultados

- RMSE típico: 1.0-1.3 con CPA
- Soporte para 18,211 genes
- Predicciones en formato Kaggle

## Tecnologías

- Python 3.9+
- LightGBM
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn


## Instalación 


### 1. Clonar el Repositorio
  
- ` git clone https://github.com/ncervanteslas/Perturbaci-n-de-Celulas-Individuales-.git`
- `cd perturbacion-celulas` 

### 2. Crear Entorno Virtual 
  
 Con conda
conda create -n perturbacion python=3.9
conda activate perturbacion

 Con venv
`python -m venv venv
source venv/bin/activate  Linux/Mac
 o  
venv\Scripts\activate  # Windows `

### 3. Instalar Dependencias

`pip install -r requirements.txt`

### 4. Configurar Datos
- Crear estructura de carpetas
`mkdir -p data features models_cuda cpa_models cpa_features`

- Colocar tus archivos de datos:
 ` de_train.parquet en data/`
  ` id_map.csv en data/`



## 📁 Estructura del Proyecto

```
Perturbacion-de-Celulas-Individuales/
├── .gitignore
├── README.md
├── requirements.txt
├── SETTINGS.json
├── LICENSE
├── data/
│   ├── de_train.parquet
│   └── id_map.csv
├── scripts/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── train_cpa.py
│   ├── predict.py
│   └── predict_cpa.py
```
