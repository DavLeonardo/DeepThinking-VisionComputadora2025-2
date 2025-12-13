# SISTEMA DE DETECCIÓN Y SEGUIMIENTO DE VEHÍCULOS EN VIDEO
YOLOv8 + Flujo Óptico

Este repositorio contiene la implementación de un sistema de visión por computador
para el monitoreo de tráfico vehicular, desarrollado como parte del curso
Visión por Computador de la Universidad Nacional de Colombia.

El sistema integra detección de objetos con YOLOv8 y seguimiento multiobjeto
mediante flujo óptico Lucas–Kanade, permitiendo mantener la identidad de los
vehículos y contar automáticamente aquellos que cruzan una línea virtual.

----------------------------------------------------------------------
OBJETIVO
----------------------------------------------------------------------

Diseñar e implementar un pipeline de análisis de video que permita:
- Detectar vehículos en cada frame
- Mantener la identidad de los objetos en el tiempo
- Contar vehículos mediante cruce de una línea virtual
- Analizar cuantitativamente el flujo de tráfico

----------------------------------------------------------------------
ARQUITECTURA DEL SISTEMA
----------------------------------------------------------------------

El sistema se divide en tres etapas principales:

ETAPA 1 – DETECCIÓN DE OBJETOS
- Modelo: YOLOv8 (yolov8n.pt)
- Dataset: COCO (preentrenado)
- Clases de interés: car, motorcycle, bus, truck
- Salidas:
  - Video con detecciones
  - Archivo JSON con bounding boxes, clases y confianza

ETAPA 2 – SEGUIMIENTO DE OBJETOS
- Flujo óptico Lucas–Kanade (Pyramidal)
- Extracción de puntos con goodFeaturesToTrack
- Asociación detección–track usando IoU
- Asignación de IDs persistentes
- Conteo de vehículos al cruzar una línea virtual

ETAPA 3 – ANÁLISIS Y VISUALIZACIÓN
- Exportación de eventos a CSV
- Análisis de:
  - Distribución por tipo de vehículo
  - Posición de cruce (carriles)
  - Flujo de tráfico en el tiempo

----------------------------------------------------------------------
REQUISITOS
----------------------------------------------------------------------

- Python 3.10 o superior
- Git
- Git LFS (para manejar videos >100MB)
- ffmpeg (recomendado para compatibilidad de video)

----------------------------------------------------------------------
INSTALACIÓN
----------------------------------------------------------------------

OPCIÓN A – GOOGLE COLAB (RECOMENDADO)

1. Abrir el notebook en Google Colab.
2. Montar Google Drive cuando se solicite.
3. Verificar las rutas:
   - VIDEO_PATH
   - RESULTS_DIR
4. Ejecutar las celdas en orden (Etapa 1 → Etapa 2 → Análisis).

Si es necesario instalar dependencias:
pip install -r requirements.txt

OPCIÓN B – EJECUCIÓN LOCAL

1. Crear entorno virtual:
python3 -m venv .venv
source .venv/bin/activate

2. Instalar dependencias:
pip install --upgrade pip
pip install -r requirements.txt

3. Instalar ffmpeg (opcional):
Arch Linux:
sudo pacman -S ffmpeg

Ubuntu/Debian:
sudo apt-get install ffmpeg

----------------------------------------------------------------------
EJECUCIÓN DEL PIPELINE
----------------------------------------------------------------------

PASO 0 – PREPARAR DATOS
Colocar el video en la carpeta:
Data/Brasil1.mp4

Crear carpeta de resultados:
Results/

ETAPA 1 – DETECCIÓN CON YOLOv8
Ejecutar el notebook de detección.

Salidas esperadas:
- Results/Brasil1_yolo.mp4
- Results/detections.json

Parámetros clave:
- CONF_THRES = 0.35
- IOU_NMS = 0.45

ETAPA 2 – TRACKING Y CONTEO
Ejecutar el notebook de seguimiento.

Salidas esperadas:
- Results/Brasil1_tracked.mp4
- Results/Brasil1_events.csv

Parámetros clave:
- LINE_Y = 300
- IOU_THRESHOLD = 0.3
- MAX_LOST = 10
- MAX_CORNERS = 20

ETAPA 3 – ANÁLISIS Y DASHBOARD
Usando el archivo Brasil1_events.csv se generan:
- Distribución por tipo de vehículo
- Posición de cruce
- Flujo de tráfico en el tiempo

----------------------------------------------------------------------
NOTAS
----------------------------------------------------------------------

- Para evitar detecciones irrelevantes, se filtran únicamente clases de vehículos.
- El conteo se realiza una sola vez por ID para evitar doble conteo.
- El uso de Git LFS es obligatorio para subir videos grandes al repositorio.

----------------------------------------------------------------------
REPORTE DE CONTRIBUCIÓN INDIVIDUAL
----------------------------------------------------------------------

Este proyecto fue desarrollado íntegramente por el autor del repositorio,
incluyendo el diseño del pipeline, implementación, experimentación,
análisis de resultados y documentación.

----------------------------------------------------------------------
