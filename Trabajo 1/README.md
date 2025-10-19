# Trabajo #1 - Vision por computadora 2025-2
Este carpeta agrupa notebooks que cubren: calibración de cámara, transformaciones de intensidad por pixel, rotaciones/traslaciones y creación de GIF, análisis y ecualización de histogramas, y segmentación por color.

## Requisitos (sistema y librerías)
- Sistema operativo: Linux / macOS / Windows (se recomiendan Linux o WSL para reproducibilidad).
- Python: versión 3.9 a 3.11.
- Entorno recomendado: Jupyter Notebook / JupyterLab (o Google Colab).
- Dependencias: requirements.txt

## Instalacion del entorno
1. Crear entorno virtual
```
python3 -m venv .venv
```
3. Activar el entorno
```
source .venv/bin/activate    # Linux/macOS
```
```
.venv\Scripts\Activate.ps1   # Windows PowerShell
```
5. Instalar dependencias
```
pip install --upgrade pip
pip install -r requirements.txt
```
## Instrucciones generales de ejecución
1. Activa el entorno virtual.
2. Verifica que las imágenes necesarias estén en las carpetas 
    - data/Imagenes_Fachada
    - data/Imágenes_Calibración
    - data/Images_P5.
3. Ejecuta el notebook o el script correspondiente a cada punto.

## P1 — Calibración de cámaras
- Objetivo: Calcular matriz intrínseca, coeficientes de distorsión y error RMS del proceso de calibración.
Ruta imágenes: ```data/Imágenes_Calibración/```

- Salida:
Matriz de cámara mtx, coeficientes dist y error RMS.
Imágenes con esquinas detectadas (cv2.drawChessboardCorners()).

## P2 — Transformaciones de intensidad por píxel
- Objetivo: Aplicar operaciones de brillo, contraste, corrección gamma y combinaciones aritméticas entre imágenes diurna/nocturna.
Ruta imágenes: ```data/Imagenes_Fachada/```
- Salida:
Imágenes ajustadas y combinadas.
Resultados comparativos de intensidad.

## P3 — Rotaciones, traslaciones y GIF
- Objetivo: Aplicar transformaciones geométricas (traslación, rotación, escala) y generar animación GIF.
Ruta imagen: ```data/Imagenes_Fachada/Img3(SELECCIONADA).jpg```
- Salida:
Carpeta outputs/Frames_Gif/ con los cuadros.
GIF generado en outputs/Gifs/.

## P4 — Distribución de intensidades y ecualización
- Objetivo: Calcular histogramas, ecualización manual y automática (cv2.equalizeHist) y comparar métricas.
Ruta imágenes: ```data/Imagenes_Fachada/```
- Salida:
Gráficas de histograma.
Imágenes ecualizadas.
Métricas de contraste, media y entropía.

## P5 — Segmentación por color (HSV)
- Objetivo: Detectar y segmentar objetos de diferentes colores, contar objetos y calcular área total.
Ruta imagen: ```data/Images_P5/Escena.jpg```
- Salida:
Máscaras por color (verde, azul, rosado, etc.).
Conteo de objetos y área total segmentada.
