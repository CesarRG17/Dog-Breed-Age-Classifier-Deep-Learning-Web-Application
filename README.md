# 🐶 Dog Breed & Age Classifier – Deep Learning Web Application

Aplicación de **visión por computadora basada en Deep Learning** que identifica la **raza** y estima el **rango de edad** de un perro a partir de una imagen.  
El proyecto está pensado como el núcleo de una **aplicación web**, utilizando un modelo multi-tarea entrenado con **PyTorch** y **Transfer Learning**.

El sistema se enfoca en **razas comunes en la Ciudad de México** y fue desarrollado con fines académicos y profesionales.

---

## 🚀 Demo
🚧 En desarrollo

---

## 🎯 Características

- Clasificación automática de **raza del perro**
- Estimación del **rango de edad**
  - 0–1 años  
  - 1–4 años  
  - 4–8 años  
  - 8–13 años
- Modelo **multi-salida (multi-task learning)**
- Transfer Learning con **ResNet**
- Dataset balanceado
- Código modular y escalable
- Preparado para integración web

---

## 🧠 Descripción del Modelo

- **Framework**: PyTorch
- **Backbone**: ResNet (preentrenada en ImageNet)
- **Salidas del modelo**:
  - Clasificación de raza (27 clases)
  - Clasificación de edad (4 clases)
- **Optimizador**: Adam
- **Función de pérdida**: CrossEntropyLoss (por tarea)

---

## 🐕 Razas Incluidas

- Akita Inu  
- Beagle  
- Border Collie  
- Boxer  
- Bulldog Francés  
- Bulldog Inglés  
- Calupoh  
- Caniche (Poodle)  
- Chihuahua  
- Cocker Spaniel  
- Dachshund  
- Dóberman  
- Golden Retriever  
- Gran Danés  
- Husky Siberiano  
- Labrador Retriever  
- Pastor Alemán  
- Pitbull Terrier  
- Pomerania  
- Pug (Carlino)  
- Rottweiler  
- San Bernardo  
- Schnauzer Miniatura  
- Shar Pei  
- Shih Tzu  
- Xoloitzcuintle  
- Yorkshire Terrier  

---

## 📁 Estructura del Dataset

```text
IMAGENES/
 ├── Beagle/
 │   ├── 0-1 anos/
 │   │   ├── Beagle01_1.jpg
 │   │   ├── Beagle01_2.jpg
 │   │   └── ...
 │   ├── 1-4 anos/
 │   ├── 4-8 anos/
 │   └── 8-13 anos/
 └── ...
