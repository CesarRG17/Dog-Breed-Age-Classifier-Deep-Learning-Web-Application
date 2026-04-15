#  Mexican Dog Breed & Age Classifier (Multi-Task Learning)

Este proyecto implementa una solución de visión artificial de extremo a extremo para la clasificación de **27 razas de perros mexicanas e internacionales**, junto con una estimación de edad mediante aprendizaje ordinal. El sistema alcanza un **97.6% de Accuracy** en la identificación de razas.

##  Valor Agregado
A diferencia de los clasificadores genéricos, este modelo:
1. **Enfoque Regional:** Especializado en razas autóctonas como el **Xoloitzcuintle** y el **Calupoh**.
2. **Arquitectura Multi-Salida:** Una sola pasada de inferencia predice simultáneamente Raza y Edad, optimizando el uso de recursos (computacionalmente eficiente).
3. **Pérdida Ordinal:** Implementación personalizada de `OrdinalCrossEntropyLoss` para que el modelo entienda la progresión biológica de la edad.

##  Arquitectura Técnica
El corazón del sistema es una **EfficientNet-B3** pre-entrenada, modificada con una capa de **GeM Pooling (Generalized Mean Pooling)** para extraer características morfológicas críticas (textura del pelaje, forma del hocico) que el Global Average Pooling estándar suele ignorar.

### Especificaciones del Modelo:
* **Backbone:** EfficientNet-B3 (Transfer Learning).
* **Cabezal de Raza:** 27 clases (Softmax).
* **Cabezal de Edad:** 4 rangos (0-1, 1-4, 4-8, 8-13 años) con lógica ordinal.
* **Regularización:** Dropout (0.4), BatchNorm, y Mixup Augmentation (alpha=0.2).

##  Desempeño y Métricas
El modelo fue evaluado rigurosamente tras un entrenamiento de 35 épocas (5 Warm-up + 30 Fine-tuning) con un dataset de **5,400 imágenes**.

### Reporte de Clasificación (Resumen):
| Tarea | Accuracy | F1-Score | Notas |
| :--- | :---: | :---: | :--- |
| **Identificación de Raza** | **97.6%** | **0.98** | Excelente desempeño en razas morfológicamente distintas. |
| **Estimación de Edad** | **59.4%** | **0.60** | Alta precisión en cachorros; reto en la distinción de etapas adultas. |

> **Análisis de Ingeniería:** Se logró un Top-3 Accuracy de **99.6%** en razas, lo que garantiza que la opción correcta esté prácticamente siempre entre las primeras sugerencias.

##  Tecnologías Utilizadas
* **Core:** `PyTorch`, `Torchvision`
* **Procesamiento:** `Pandas`, `NumPy`, `Scikit-learn`
* **API / Backend:** `FastAPI` (Inferencia asíncrona)
* **Frontend:** `Tailwind CSS`, `JavaScript (Fetch API)`
* **Métricas:** `Seaborn` & `Matplotlib` (Matrices de Confusión y Curvas de Pérdida)

##  Pipeline de Datos
1. **Pre-procesamiento:** Redimensionamiento a 224x224, Normalización Imagenet.
2. **Aumentación:** RandomResizedCrop, ColorJitter, RandomErasing (para robustez ante oclusiones).
3. **Inferencia:** Una vez predicha la raza y edad, el sistema consulta un archivo `Pesos_Final.csv` estandarizado para devolver el peso promedio del ejemplar.

##  Cómo ejecutar el proyecto
1. Instala las dependencias: `pip install -r requirements.txt`
2. Ejecuta el backend: `python backend.py`
3. Abre `index.html` para realizar pruebas de inferencia.

---
**Proyecto desarrollado como parte de la formación en Ingeniería de Inteligencia Artificial.**
