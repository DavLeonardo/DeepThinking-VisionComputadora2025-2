# Registro y Medición Métrica de Imágenes — Guía de Uso (README)

¡Bienvenido! Este proyecto implementa un **pipeline completo** para:

1) **Validar** con datos sintéticos el registro de imágenes.
2) **Unir** (stitching) múltiples fotos reales de un mismo escenario.
3) **Calibrar y medir** distancias reales en la imagen final usando referencias (cuadro y mesa).

> Recomendado para personas con **poca experiencia** técnica: sigue los pasos tal cual y revisa la sección de **Solución de problemas** si algo no abre.

---

## 1) ¿Qué puedes hacer con este proyecto?

- Generar un **mosaico** de 2–3 fotos de una misma escena (p. ej., un comedor).
- **Calibrar la escala** (píxeles → centímetros) usando:
  - **Altura del cuadro**: 117.0 cm
  - **Ancho de la mesa**: 161.1 cm
- **Medir** objetos en la imagen final con una interfaz **interactiva** (dar clic en dos puntos para medir).

---

## 2) Estructura del repositorio

```
src/
 ├── feature_detection.py     # Detecta puntos (SIFT, ORB, AKAZE)
 ├── matching.py              # Empareja descriptores + filtro ratio (Lowe)
 ├── registration.py          # Estima homografías (RANSAC) + métricas
 ├── stitching.py             # Une imágenes con homografía (stitching)
 ├── measurement.py           # Calibración y medición interactiva
 └── __init__.py
notebooks/
 ├── 02_synthetic_validation.ipynb  # Validación con datos sintéticos
 ├── 03_main_pipeline.ipynb         # Pipeline real (registro / mosaico)
 └── 04_measurement.ipynb           # Calibración + mediciones interactivas
data/
 ├── original/   # (Coloca aquí tus imágenes reales)
 └── synthetic/  # (Se usa en la validación sintética)
results/
 ├── figures/        # Salidas gráficas (mosaicos, comparativas, etc.)
 └── measurements/   # CSV y JPG con puntos/lineas de medición
requirements.txt
README.md  ← este archivo
```

---

## 3) Requisitos del sistema

- **Python** 3.10 o 3.11
- **Espacio en disco** ~2 GB libres (para dependencias y resultados)
- Sistema operativo: Windows / macOS / Linux

### Paquetes gráficos (para la ventana interactiva de medición)

La medición abre una **ventana** donde harás clic sobre el mosaico. Para eso se usa *Matplotlib* con un **backend gráfico**:

- Opción A (**recomendada**): **Qt5** → se instala automáticamente con **PyQt5** (incluido en `requirements.txt`).
- Opción B (alternativa): **Tk** → según el sistema, puede requerir un paquete del sistema (ver sección 7).

> Si no puedes instalar componentes gráficos (por ejemplo, en servidores remotos sin interfaz), **NO** podrás usar la herramienta de medición interactiva. En ese caso, abre los notebooks desde una máquina con escritorio gráfico.

---

## 4) Instalación paso a paso (la forma más simple)

### 4.1. Crear un entorno (opción 1: con `venv` estándar)

```bash
# 1) Crea y activa un entorno (Windows)
python -m venv .venv
.venv\Scripts\activate

# 1) Crea y activa un entorno (macOS / Linux)
python3 -m venv .venv
source .venv/bin/activate
```

### 4.2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota sobre OpenCV (SIFT)**: usamos `opencv-contrib-python` (incluye SIFT). Si instalaste por error `opencv-python` (sin *contrib*), desinstala e instala de nuevo:

```bash
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

### 4.3. Verificar instalación de OpenCV y SIFT

```bash
python - << "PY"
import cv2
print("OpenCV:", cv2.__version__)
try:
    _ = cv2.SIFT_create()
    print("SIFT OK ✅ (opencv-contrib instalado)")
except Exception as e:
    print("SIFT NO DISPONIBLE ❌ — instala opencv-contrib-python:", e)
PY
```

### 4.4. Instalar Jupyter (ya viene en requirements)

Lanzar notebooks:

```bash
jupyter notebook
```

Se abrirá tu navegador web. Entra a `notebooks/` y abre los cuadernos en este orden:

1. `02_synthetic_validation.ipynb` (opcional, validación)
2. `03_main_pipeline.ipynb` (genera el mosaico real)
3. `04_measurement.ipynb` (calibración + medición interactiva)

---

## 5) Cómo ejecutar el **pipeline de registro** (imágenes reales)

1. Copia tus fotos en `data/original/` (por ejemplo, `img1.jpg`, `img2.jpg`, `img3.jpg`).
2. Abre `notebooks/03_main_pipeline.ipynb` y sigue las celdas (detecta → empareja → estima homografía → une).
3. El resultado (mosaico) se guardará en `results/figures/` (o en la ruta que el notebook especifique).

> **Consejo**: si el solape entre fotos es pequeño o hay cambios de iluminación, prueba otro detector (SIFT/ORB/AKAZE) y ajusta el umbral del **ratio test**.

---

## 6) Cómo ejecutar la **medición interactiva** (¡abre ventana!)

Tienes dos caminos:

### 6.1. (Recomendado) Usar el notebook

1. Abre `notebooks/04_measurement.ipynb`
2. Ejecuta las celdas.
3. Cuando aparezca la **ventana** con la imagen:
   - Haz clic **1 y 2** sobre los extremos verticales del **cuadro** (altura real: 117.0 cm).
   - Haz clic **3 y 4** sobre los extremos del **ancho de la mesa** (161.1 cm).
   - Para **cada objeto** que quieras medir: haz clic **dos puntos** (origen y fin).
   - Cierra la ventana para terminar.
4. Se guardarán:
   - Imagen anotada: `results/measurements/panorama_con_puntos.jpg`
   - CSV de resultados: `results/measurements/mediciones.csv`

### 6.2. (Alternativa) Llamarlo desde la terminal (avanzado)

Asegúrate de que la ruta del mosaico sea correcta:

```bash
python - << "PY"
from src.measurement import medir_interactivo
# Cambia esta ruta por tu mosaico real
df, out_img, out_csv = medir_interactivo("results/figures/panorama.jpg")
print("Listo ✅")
print("Imagen anotada:", out_img)
print("CSV:", out_csv)
print(df.head())
PY
```

> **Importante**: La medición **requiere** una ventana gráfica. Si estás en un servidor remoto sin interfaz, no funcionará.

---

## 7) Requisitos gráficos por sistema (solo para la medición)

La herramienta intenta usar primero **Qt5**, y si no está disponible, usa **Tk** automáticamente.

### Windows

- Con `pip install -r requirements.txt`, **PyQt5** queda instalado y suele funcionar sin pasos extra.
- Si la ventana **no abre**, prueba:
  1. Cerrar y volver a abrir la consola/terminal.
  2. Ejecutar: `python -m pip install --upgrade PyQt5`

### macOS

- Recomendado: usar **Qt (PyQt5)** incluido en `requirements.txt`.
- Alternativa (Tk): instalar con *Homebrew*:
  ```bash
  brew install python-tk
  ```

### Linux (Ubuntu/Debian)

- Recomendado: **PyQt5** (ya en `requirements.txt`).
- Alternativa (Tk):
  ```bash
  sudo apt update
  sudo apt install python3-tk
  ```

### Linux (Arch/Manjaro)

- Recomendado: **PyQt5** (ya en `requirements.txt`).
- Alternativa (Tk):
  ```bash
  sudo pacman -S tk
  ```

> Si aparece el error **"Qt platform plugin could not be initialized"**:
>
> - En Windows: reinstala PyQt5 (`pip install --force-reinstall PyQt5`).
> - En Linux/macOS: revisa variables de entorno o intenta con **Tk** (instala `python3-tk` / `tk`).

---

## 8) Solución de problemas (FAQ)

**Q1. El cuaderno de medición corre pero NO aparece la ventana.**

- Causa: estás en un entorno sin interfaz gráfica (servidor remoto) o el backend no está disponible.
- Solución: ejecuta en un PC con escritorio; instala **PyQt5** o **Tk** (ver sección 7).

**Q2. Error: `cv2` no encontrado o SIFT no disponible.**

- Instala **opencv-contrib-python** (ver sección 4.2).

**Q3. El mosaico sale en blanco o muy oscuro.**

- Verifica el orden de imágenes y que haya **solape suficiente**.
- Prueba otro detector (SIFT/ORB/AKAZE) y ajusta el **ratio test** en `matching.py`.

**Q4. Pude medir, pero quiero **promediar** la escala de cuadro y mesa.**

- La herramienta guarda ambas referencias. Puedes calcular una escala combinada (promedio) en el CSV resultante o extender `measurement.py` para hacerlo automáticamente.

**Q5. ¿Puedo ejecutar todo sin notebooks?**

- Sí, pero es más complejo. Recomendamos los notebooks para evitar errores y ver los pasos.

---

## 9) Créditos y referencias

- D. Lowe, **Distinctive Image Features from Scale-Invariant Keypoints (SIFT)**, 2004.
- Hartley & Zisserman, **Multiple View Geometry**.
- **OpenCV**: documentación oficial de *features2d*, *calib3d* y *stitching*.
- Documentación de **Matplotlib** y **PyQt5**.

---

## 10) Resumen

1) Instala requisitos → 2) Ejecuta `03_main_pipeline.ipynb` para crear el mosaico → 3) Ejecuta `04_measurement.ipynb` para medir con la ventana interactiva.
   Los resultados se guardan en `results/figures/` y `results/measurements/`.

¡Éxitos y buenas mediciones! 🔎📏
