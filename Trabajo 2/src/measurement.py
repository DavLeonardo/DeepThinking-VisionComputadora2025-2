# src/measurement.py
"""
Parte 3: Calibración y medición con ginput, estilo del notebook original.
- Backend preferido: Qt5Agg (rápido, estable). Fallback: TkAgg.
- Clics:
    (1,2) -> altura del cuadro = 117 cm
    (3,4) -> ancho de la mesa  = 161.1 cm
    (pares siguientes) -> objetos a medir
- Salidas:
    - JPG anotado: results/measurements/panorama_con_puntos.jpg
    - CSV:         results/measurements/mediciones.csv
"""

import os
import math
from typing import Tuple, List

import cv2
import numpy as np
import pandas as pd
import matplotlib

# --- Selección de backend interactivo ---
# Preferimos Qt5Agg; si no se puede, probamos TkAgg. Si no, advertimos.
_backend = matplotlib.get_backend().lower()
if _backend in ("agg", "module://matplotlib_inline.backend_inline"):
    _set = False
    try:
        matplotlib.use("Qt5Agg")  # requiere: sudo pacman -S python-pyqt5
        _set = True
    except Exception as e:
        print("[measurement] Aviso: no se pudo activar Qt5Agg:", e)
    if not _set:
        try:
            matplotlib.use("TkAgg")  # requiere: sudo pacman -S tk
            _set = True
        except Exception as e:
            print("[measurement] Aviso: tampoco se pudo activar TkAgg:", e)
    if not _set:
        print(
            "[measurement] Advertencia: Matplotlib sigue en modo no interactivo; ginput no abrirá ventana."
        )

import matplotlib.pyplot as plt


def _dist_px(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Distancia euclidiana en píxeles."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _sigma_cm(escala_cm_px: float, sigma_px: float = 2.0) -> float:
    """Propaga ±2 px a centímetros con la escala (cm/px)."""
    return abs(escala_cm_px * sigma_px)


def medir_interactivo(
    panorama_path: str,
    altura_cuadro_cm: float = 117.0,
    ancho_mesa_cm: float = 161.1,
    out_dir: str = "results/measurements",
):
    """
    Abre el panorama, solicita clics con ginput y guarda:
      - JPG anotado con puntos y líneas
      - CSV con calibraciones y mediciones

    Retorna
    -------
    (df, ruta_imagen_salida, ruta_csv_salida)
    """
    os.makedirs(out_dir, exist_ok=True)

    img_bgr = cv2.imread(panorama_path)
    if img_bgr is None:
        raise FileNotFoundError(
            f"No se pudo cargar el panorama: {os.path.abspath(panorama_path)}"
        )
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Mostrar imagen y solicitar clics (cierra la ventana para terminar)
    plt.figure(figsize=(12, 6))
    plt.title(
        "Selecciona:\n1–2 cuadro (117 cm), 3–4 mesa (161.1 cm), luego pares extra → cierra la ventana"
    )
    plt.imshow(img)
    plt.axis("off")
    pts: List[Tuple[float, float]] = plt.ginput(n=-1, timeout=0)
    plt.close()

    if len(pts) < 4:
        raise RuntimeError(
            "Se requieren al menos 4 puntos (2 para cuadro, 2 para mesa)."
        )

    # Calibraciones
    px_cuadro = _dist_px(pts[0], pts[1])
    px_mesa = _dist_px(pts[2], pts[3])
    esc_pared = altura_cuadro_cm / px_cuadro  # cm/px
    esc_mesa = ancho_mesa_cm / px_mesa  # cm/px

    # Tabla inicial (trazabilidad)
    regs = [
        {
            "etiqueta": "calib_cuadro",
            "px": px_cuadro,
            "cm_pared": altura_cuadro_cm,
            "cm_mesa": px_cuadro * esc_mesa,
            "σ_pared_cm": np.nan,
            "σ_mesa_cm": np.nan,
        },
        {
            "etiqueta": "calib_mesa",
            "px": px_mesa,
            "cm_pared": px_mesa * esc_pared,
            "cm_mesa": ancho_mesa_cm,
            "σ_pared_cm": np.nan,
            "σ_mesa_cm": np.nan,
        },
    ]

    # Dibujo + medición de pares extra
    img_vis = img.copy()
    for i, p in enumerate(pts):
        cv2.circle(img_vis, (int(p[0]), int(p[1])), 6, (0, 255, 0), -1)
        cv2.putText(
            img_vis,
            str(i + 1),
            (int(p[0]) + 6, int(p[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    extras = pts[4:]
    for i in range(0, len(extras), 2):
        if i + 1 >= len(extras):
            break
        p1, p2 = extras[i], extras[i + 1]
        Lpx = _dist_px(p1, p2)
        regs.append(
            {
                "etiqueta": f"extra_{(i//2)+1}",
                "px": Lpx,
                "cm_pared": Lpx * esc_pared,
                "cm_mesa": Lpx * esc_mesa,
                "σ_pared_cm": _sigma_cm(esc_pared),
                "σ_mesa_cm": _sigma_cm(esc_mesa),
            }
        )
        cv2.line(
            img_vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 3
        )

    # Guardar salidas
    out_img = os.path.join(out_dir, "panorama_con_puntos.jpg")
    cv2.imwrite(out_img, cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
    df = pd.DataFrame(regs)
    out_csv = os.path.join(out_dir, "mediciones.csv")
    df.to_csv(out_csv, index=False)

    return df, out_img, out_csv
