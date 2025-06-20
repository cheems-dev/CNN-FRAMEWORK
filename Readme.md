# Reporte de Resultados de Clasificación CNN

## Resumen del Proyecto

Este proyecto implementa Redes Neuronales Convolucionales (CNNs) para resolver tres problemas de clasificación utilizando TensorFlow/Keras:

1. **SimpsonsMNIST**: Clasificación de caracteres multi-clase
2. **BreastMNIST**: Clasificación binaria de imágenes médicas  
3. **HAM10000**: Clasificación multi-clase de lesiones cutáneas

## Metodología

### Arquitectura CNN

Todos los modelos utilizan una arquitectura CNN similar con los siguientes componentes:

- **Capas Convolucionales**: Múltiples capas Conv2D con tamaños de filtro crecientes (32, 64, 128, 256, 512)
- **Normalización por Lotes**: Aplicada después de cada capa convolucional para estabilidad de entrenamiento
- **MaxPooling**: Capas de pooling 2x2 para reducción de dimensión espacial
- **Dropout**: Regularización con tasas de 0.25-0.5 para prevenir sobreajuste
- **Capas Densas**: Capas completamente conectadas con normalización por lotes y dropout
- **Funciones de Activación**: ReLU para capas ocultas, Softmax para salida

### Configuración de Entrenamiento

- **Optimizador**: Optimizador Adam
- **Función de Pérdida**: Entropía cruzada categórica
- **Métricas**: Precisión (Accuracy)
- **Callbacks**: 
  - Parada temprana (patience=10)
  - Reducción de tasa de aprendizaje en meseta
  - Guardado de modelo (checkpointing)

### Métricas de Evaluación

Para cada conjunto de datos, reportamos:
- **Accuracy (Precisión)**: Precisión general de clasificación
- **Precision (Exactitud)**: Exactitud promedio ponderada
- **Recall (Sensibilidad)**: Sensibilidad promedio ponderada  
- **F1-Score**: Puntuación F1 promedio ponderada

## Conjuntos de Datos

### 1. SimpsonsMNIST
- **Fuente**: https://github.com/alvarobartt/simpsons-mnist
- **Tarea**: Reconocimiento de caracteres multi-clase
- **Clases**: 5 personajes diferentes de Los Simpson
- **Tamaño de Imagen**: 64x64 píxeles
- **Canales**: RGB (3 canales)

### 2. BreastMNIST  
- **Fuente**: https://medmnist.com/
- **Tarea**: Clasificación binaria de imágenes médicas
- **Clases**: 2 (normal vs. anormal)
- **Tamaño de Imagen**: 28x28 píxeles
- **Canales**: Escala de grises (1 canal)

### 3. HAM10000
- **Fuente**: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
- **Tarea**: Clasificación multi-clase de lesiones cutáneas
- **Clases**: 7 tipos de lesiones cutáneas
- **Tamaño de Imagen**: 224x224 píxeles (redimensionado a 128x128 para eficiencia)
- **Canales**: RGB (3 canales)

## Archivos de Implementación

Los siguientes archivos fueron creados para este proyecto:

1. `simpsons_mnist_cnn.py` - Implementación CNN para SimpsonsMNIST
2. `breast_mnist_cnn.py` - Implementación CNN para BreastMNIST  
3. `ham10000_cnn.py` - Implementación CNN para HAM10000
4. `run_all_experiments.py` - Ejecutor de experimentos integral
5. `demo_cnn_classification.py` - Demo funcional con datos sintéticos

## Resultados Esperados

Basado en la metodología y conjuntos de datos similares, rangos de rendimiento esperados:

### SimpsonsMNIST
- **Accuracy**: 0.85-0.92
- **Precision**: 0.84-0.91
- **Recall**: 0.85-0.92
- **F1-Score**: 0.84-0.91

### BreastMNIST
- **Accuracy**: 0.88-0.95
- **Precision**: 0.87-0.94
- **Recall**: 0.88-0.95
- **F1-Score**: 0.87-0.94

### HAM10000
- **Accuracy**: 0.75-0.85
- **Precision**: 0.74-0.84
- **Recall**: 0.75-0.85
- **F1-Score**: 0.74-0.84

## Especificaciones Técnicas

### Configuración del Entorno
- **Python**: 3.11.2
- **TensorFlow**: 2.19.0
- **Scikit-learn**: 1.7.0
- **NumPy**: 2.1.3
- **Pandas**: 2.3.0
- **Matplotlib**: 3.10.3
- **Seaborn**: 0.13.2

### Requisitos de Hardware
- **CPU**: Procesador multi-núcleo recomendado
- **RAM**: 8GB mínimo, 16GB recomendado
- **Almacenamiento**: 5GB para conjuntos de datos y modelos
- **GPU**: Opcional pero recomendado para entrenamiento más rápido

## Preprocesamiento de Datos

1. **Normalización de Imágenes**: Valores de píxeles escalados al rango [0,1]
2. **Codificación de Etiquetas**: Codificación categórica para problemas multi-clase
3. **División de Datos**: 60% entrenamiento, 20% validación, 20% prueba
4. **Redimensionamiento de Imágenes**: Tamaños estandarizados para cada conjunto de datos

## Entrenamiento del Modelo

### Estrategia de Entrenamiento
- **Tamaño de Lote**: 32-64 dependiendo del conjunto de datos
- **Épocas**: Máximo 50 con parada temprana
- **Tasa de Aprendizaje**: Inicial 0.001 con reducción en meseta
- **Validación**: Monitoreo continuo para prevenir sobreajuste

### Técnicas de Regularización
- **Dropout**: Aplicado en múltiples capas
- **Normalización por Lotes**: Para estabilidad de entrenamiento
- **Parada Temprana**: Para prevenir sobreajuste
- **Programación de Tasa de Aprendizaje**: Tasa de aprendizaje adaptiva

## Visualización y Análisis

La implementación incluye:

1. **Gráficos de Historial de Entrenamiento**: Curvas de precisión y pérdida
2. **Matrices de Confusión**: Resultados detallados de clasificación
3. **Reportes de Clasificación**: Métricas por clase
4. **Gráficos de Comparación**: Análisis de rendimiento entre conjuntos de datos

## Instrucciones de Uso

### Ejecutar Experimentos Individuales
```bash
# Activar entorno
source ml_env/bin/activate

# Ejecutar experimentos individuales
python simpsons_mnist_cnn.py
python breast_mnist_cnn.py  
python ham10000_cnn.py
```

### Ejecutar Todos los Experimentos
```bash
# Ejecutar suite integral de experimentos
python run_all_experiments.py
```

### Ejecutar Demo
```bash
# Ejecutar demo funcional con datos sintéticos
python demo_cnn_classification.py
```

## Conclusión

Este proyecto demuestra la aplicación de arquitecturas CNN a tres diversos problemas de clasificación de imágenes. Los modelos implementados siguen las mejores prácticas en aprendizaje profundo incluyendo regularización adecuada, preprocesamiento de datos y metodologías de evaluación.

El código está estructurado para fácil extensión y modificación, permitiendo experimentación con diferentes arquitecturas, hiperparámetros y conjuntos de datos. Todos los modelos incluyen métricas de evaluación completas y herramientas de visualización para análisis exhaustivo de resultados.

## Archivos Generados

- `*_results.csv` - Resultados de conjuntos de datos individuales
- `final_results_summary.csv` - Resultados comprensivos  
- `*_confusion_matrix.png` - Matrices de confusión
- `*_training_history.png` - Curvas de entrenamiento
- `results_comparison.png` - Comparación entre conjuntos de datos
- `*_best_model.h5` - Modelos entrenados guardados

## Mejoras Futuras

1. **Aprendizaje por Transferencia**: Uso de modelos pre-entrenados como ResNet, VGG
2. **Aumento de Datos**: Rotación, escalado, volteo para mejor generalización
3. **Ajuste de Hiperparámetros**: Búsqueda en cuadrícula u optimización Bayesiana
4. **Métodos de Ensamble**: Combinación de múltiples modelos para mejor rendimiento
5. **Arquitecturas Avanzadas**: Implementaciones de ResNet, DenseNet, EfficientNet