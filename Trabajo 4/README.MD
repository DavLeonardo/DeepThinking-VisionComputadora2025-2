# Monitoreo de Tráfico en Video: YOLOv8 + Tracking (Lucas–Kanade)

Este proyecto implementa un sistema de:
1) **Detección de vehículos** con **YOLOv8**  
2) **Seguimiento multiobjeto** con **flujo óptico Lucas–Kanade** + corrección por **IoU**  
3) **Conteo por cruce de línea** y exportación de eventos a **CSV**  
4) **Dashboard** con distribución por clase, posición de cruce y flujo en el tiempo.

---

## ✅ Requisitos

- Python 3.10+ (recomendado)
- `pip`
- (Opcional pero recomendado) `ffmpeg` para convertir el video final a H.264

---

## 📦 Instalación

### Opción A — Google Colab (recomendado)
1. Abre el notebook en Colab.
2. Monta Drive cuando se solicite.
3. Verifica/ajusta rutas:
   - `VIDEO_PATH`
   - `RESULTS_DIR`
4. Ejecuta las celdas en orden (Etapa 1 → Etapa 2 → Dashboard).

> En Colab normalmente ya están `numpy`, `opencv` y `pandas`, pero si algo falla ejecuta:
```bash
pip install -r requirements.txt
