# Perturbación de Células Individuales - Pipeline de Predicción Génica

Pipeline completo para predecir expresión génica en respuesta a perturbaciones químicas, implementando modelos avanzados de machine learning incluyendo CPA (Compositional Perturbation Autoencoder) y embeddings de moléculas con ChemBERTa.

## 🎯 Características

- **CPA (Compositional Perturbation Autoencoder)**: Modelo especializado que captura interacciones complejas
- **ChemBERTa**: Representaciones de moléculas usando transformers pre-entrenados
- **LightGBM con CUDA**: Entrenamiento acelerado por GPU
- **Pipeline completo**: Desde preprocesamiento hasta predicción
- **Validación cruzada**: 5-fold estratificada

## 🚀 Comenzar

1. Clonar repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Colocar datos en `data/`
4. Ejecutar: `python scripts/prepare_data.py`
5. Entrenar: `python scripts/train_cpa.py`
6. Predecir: `python scripts/predict_cpa.py`

## 📊 Resultados

- RMSE típico: 1.0-1.3 con CPA
- Soporte para 18,211 genes
- Predicciones en formato Kaggle

## 🛠️ Tecnologías

- Python 3.9+
- LightGBM
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
