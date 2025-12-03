# Trabajo 3 – Clasificación de Imágenes Médicas

## Descriptores Clásicos vs Deep Learning

Curso: **Visión por Computador – 3009228**
Universidad Nacional de Colombia – Facultad de Minas

## 1. Descripción general

Este repositorio contiene la implementación completa del **Trabajo 3: Clasificación de Imágenes Médicas con Descriptores Clásicos vs Deep Learning**.

El objetivo principal es comparar dos enfoques para la detección de neumonía a partir de radiografías de tórax:

1. **Descriptores clásicos (handcrafted)** de forma y textura + clasificadores tradicionales.
2. **Red neuronal convolucional (CNN)** entrenada directamente sobre las imágenes.

Todo el pipeline va desde la descarga y preprocesamiento del dataset, pasando por la extracción de descriptores, hasta la etapa de clasificación y evaluación.

## 2. Dataset

Se utiliza el conjunto de datos público:

> **Chest X-Ray Images (Pneumonia)**
> Kaggle – Paul Mooney
> https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Clases:

- `NORMAL`: radiografías de tórax de pacientes sanos.
- `PNEUMONIA`: radiografías con diagnóstico de neumonía.

El dataset original presenta **alto desbalance** a favor de la clase *PNEUMONIA*, por lo que en el preprocesamiento se hace una nueva partición estratificada en Train/Val/Test y se calculan *class weights* para compensar el desbalance.

## 3. Estructura del repositorio

Una estructura sugerida es:

```text
.
├── notebooks/
│   ├── 01_parte1_preprocesamiento.ipynb
│   ├── 02_parte2A_descriptores_forma.ipynb
│   ├── 03_parte2B_descriptores_textura.ipynb
│   └── 04_parte3_clasificacion.ipynb
├── data/
│   └── Dataset/            # Imágenes preprocesadas (Train / Val / Test)
├── README.md
└── requirements.txt
```

# 4. Pipeline implementado

## 4.1 Parte 1 – Análisis exploratorio y preprocesamiento

Notebook: `01_parte1_preprocesamiento.ipynb`

* **Descarga del dataset** desde Kaggle usando `kagglehub`.
* **EDA inicial:**
  * Visualización de ejemplos de ambas clases.
  * Análisis de la distribución de clases (desbalance marcado hacia PNEUMONIA).
  * Análisis de tamaños de imagen (gran variabilidad en resolución).
* **Redefinición del split** :
* Se mezclan todas las imágenes y se crea un nuevo split estratificado:
  * Train ~ 70–75 %
  * Val ~ 15 %
  * Test ~ 10–15 %
* **Preprocesamiento aplicado:**
  * Conversión a escala de grises.
  * **CLAHE** (ecualización adaptativa de histograma) para mejorar el contraste.
  * Redimensionamiento uniforme a `256 × 256`.
  * Guardado de imágenes procesadas en una nueva carpeta `Dataset/Train`, `Dataset/Val`, `Dataset/Test`.
  * Cálculo de **pesos por clase** para usar en el entrenamiento.

## 4.2 Parte 2A – Descriptores de forma

Notebook: `02_parte2A_descriptores_forma.ipynb`

Se implementan y analizan varios descriptores clásicos de forma:

1. **Histogram of Oriented Gradients (HOG)**
   * Experimentación con:
     * `pixels_per_cell` ∈ {(8,8), (16,16), (32,32)}
     * número de bins de orientación ∈ {9, 12, 16}
   * Visualización de las imágenes HOG para entender qué bordes y estructuras están capturando.
2. **Momentos de Hu**
   * Cálculo de los 7 momentos invariantes a:
     * traslación
     * escala
     * rotación
3. **Descriptores de contorno**
   * Segmentación binaria mediante umbral de Otsu.
   * Cálculo de:
     * Área
     * Perímetro
     * Circularidad
     * Excentricidad (elipse ajustada al contorno principal)
4. **Fourier Shape Descriptors (FSD)**
   * Transformada de Fourier sobre el contorno más grande.
   * Normalización para invariancia a traslación y escala.
   * Uso de los primeros `N` coeficientes para representar la forma.

## 4.3 Parte 2B – Descriptores de textura

Notebook: `03_parte2B_descriptores_textura.ipynb`

Se implementan y estudian descriptores clásicos de textura:

1. **Local Binary Patterns (LBP)**
   * Implementación usando `skimage`.
   * Cálculo del histograma de patrones LBP.
   * Experimentación con diferentes parámetros:
     * Puntos vecinos `P` ∈ {8, 16, 24}
     * Radios `R` ∈ {1, 2, 3}
2. **Gray Level Co-occurrence Matrix (GLCM)**
   * Cálculo de matrices de co-ocurrencia para varias:
     * distancias: `d ∈ {1, 2, 3, 4}`
     * direcciones: `θ ∈ {0, π/4, π/2, 3π/4}`
   * Extracción de propiedades:
     * contraste
     * correlación
     * energía
     * homogeneidad
3. **Filtros de Gabor**
   * Banco de filtros con:
     * frecuencias (ej. 0.1, 0.2, 0.3)
     * orientaciones (0, π/4, π/2, 3π/4)
   * Para cada filtro se obtienen:
     * media
     * desviación estándar de la magnitud de la respuesta
4. **Estadísticos de primer orden**
   * Calculados directamente sobre la imagen:
     * media
     * varianza
     * skewness
     * kurtosis
     * entropía
5. **Vector de descriptores integrado**
   * Función `extract_descriptors_completo` que concatena:
     * HOG
     * Hu
     * FSD
     * LBP
     * GLCM
     * Gabor
     * Estadísticos de primer orden
   * Visualizaciones:
     * vector completo para una imagen
     * gráficos de barras por grupo de descriptor
     * PCA 2D / 3D para ver separación de clases
     * mapas de calor de correlación entre features
     * radar plots (firma espectral de los descriptores)

## 4.4 Parte 3 – Clasificación con descriptores clásicos y CNN

Notebook: `04_parte3_clasificacion.ipynb`

1. **Construcción de la matriz de características**
   A partir de las imágenes preprocesadas (`256×256`), se extraen de forma sistemática:

   * HOG
   * Momentos de Hu
   * Descriptores de contorno
   * LBP
   * GLCM
   * Gabor

   El vector final tiene alrededor de **8000+ características** por imagen.

   Se construyen matrices `X_train`, `X_val`, `X_test` y sus etiquetas correspondientes.
   Para los modelos clásicos, se combinan `Train + Val` como `X_train_full` y se reserva `Test` solo para evaluación final.
2. **Modelos clásicos de ML**
   Se usan `pipelines` con **escalado de características** (`StandardScaler`) cuando aplica:

   * SVM (kernel lineal)
   * SVM (kernel RBF)
   * Random Forest
   * k-NN (k=5)
   * Regresión logística

   Para cada modelo se calculan:

   * Accuracy
   * Precision
   * Recall
   * F1-Score
   * Reporte de clasificación por clase
   * Matriz de confusión
   * Curva ROC y AUC

   ```

   ```

   En las pruebas realizadas:

   * Los modelos logran accuracies entre ~93 % y ~96 % en el conjunto de  **test** .
   * **SVM con kernel RBF** es el modelo más fuerte, con mejor trade-off entre *precision* y *recall* para ambas clases.
   * La clase **PNEUMONIA** suele tener *recall* muy alto (detección de casos positivos), mientras que la clase **NORMAL** es la más difícil.
3. **Red convolucional (CNN)**
   Además de los modelos clásicos, se implementa un modelo CNN que trabaja directamente sobre las imágenes:

   * Generadores de datos (`ImageDataGenerator`) con augmentación:
     * rotación, traslación, zoom, flips horizontales.
   * Arquitectura basada en **SeparableConv2D** (convoluciones separables en profundidad) para reducir parámetros.
   * Callbacks:
     * `EarlyStopping`
     * `ReduceLROnPlateau`

   La CNN se entrena y se evalúa sobre los conjuntos Train/Val/Test, permitiendo comparar su comportamiento con los clasificadores basados en descriptores clásicos.

# 5. Resultados principales

De forma cualitativa:

* Todos los clasificadores clásicos alcanzan métricas altas (accuracy ≳ 0.93).
* **SVM RBF** muestra el mejor desempeño global y una muy buena detección de neumonía.
* La variabilidad de tamaños y el desbalanceo de clases hacen que el preprocesamiento (CLAHE + resize + split estratificado) sea clave para estabilizar los resultados.
* La CNN proporciona una alternativa end-to-end y permite comparar el enfoque de *feature learning* automático frente a *feature engineering* manual.

Si se desea, se puede añadir en esta sección una tabla con las métricas numéricas exactas copiadas desde las salidas del notebook.

# 6. Requisitos y ejecución

## 6.1 Instalación de dependencias

```bash
# Clonar el repositorio
git clone <URL_DE_TU_REPOSITORIO>
cd <NOMBRE_DEL_REPO>

# Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 6.2 Descarga del dataset

Tienes dos opciones:

1. **Usar los notebooks tal cual en Google Colab**
   * Subir el repositorio a tu cuenta de GitHub.
   * Abrir cada notebook en Colab.
   * Configurar las credenciales de Kaggle (`KAGGLE_KEY` / `KAGGLE_USERNAME`) o editar las rutas para usar una copia manual del dataset.
2. **Descargar manualmente desde Kaggle**
   * descargar el ZIP desde la página del dataset;
   * descomprimir en `data/chest_xray/` o en la ruta que prefieras;
   * ajustar las variables `base_path` / `BASE_PATH` en los notebooks para que apunten a esa ruta.

### 6.3 Ejecución sugerida

1. Ejecutar **Parte 1** para generar las imágenes preprocesadas y el nuevo split Train/Val/Test.
2. Ejecutar **Parte 2A** y **Parte 2B** para entender y visualizar los descriptores.
3. Ejecutar **Parte 3** para:
   * construir la matriz de características completa;
   * entrenar los modelos clásicos;
   * entrenar y evaluar la CNN;
   * generar las métricas y visualizaciones (matrices de confusión, curvas ROC, etc.).

# 7. Créditos y referencias

* **Dataset:** Paul Mooney –  *Chest X-Ray Images (Pneumonia)* , Kaggle.

```yam

### requirements.txt

kagglehub
numpy
matplotlib
seaborn
pandas
opencv-python
scikit-image
scikit-learn
scipy
tqdm
imutils
tensorflow
```
