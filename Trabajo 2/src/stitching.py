"""
stitching.py
------------
Unificación (stitching) de imágenes mediante una homografía estimada, con manejo
cuidadoso de las regiones de interés (ROI) para evitar errores de índices y artefactos
en bordes. Se utiliza una "doble máscara" para no sobreescribir contenido válido.

Función clave:
    - unificar_dos_imagenes(img_base_color, img_to_warp_color, H)
"""

import numpy as np
import cv2


def unificar_dos_imagenes(img_base_color, img_to_warp_color, H):
    """
    Proyecta 'img_to_warp_color' sobre el lienzo de 'img_base_color' usando la
    homografía H (que mapea src->dst), y retorna el mosaico en BGR.

    Parámetros
    ----------
    img_base_color : np.ndarray
        Imagen base en BGR (será el lienzo del mosaico).
    img_to_warp_color : np.ndarray
        Imagen a proyectar/encajar en el lienzo.
    H : np.ndarray
        Homografía 3x3 que transforma coordenadas de 'img_to_warp_color' a
        'img_base_color'.

    Retorna
    -------
    mosaic_bgr : np.ndarray
        Imagen mosaicada en BGR.

    Notas
    -----
    - Convierte a escala de grises para calcular dimensiones y máscaras (más simple).
    - Calcula los límites del canvas transformando las esquinas de la imagen a warp.
    - Usa una traslación para garantizar coordenadas positivas en el canvas.
    - Aplica "doble máscara" para pegar la base sin borrar contenido proyectado.
    """
    if H is None or H.shape != (3, 3):
        raise ValueError("H inválida. Debe ser una matriz 3x3.")

    # Convertimos a gris para operaciones de warp/máscara
    base_gray = cv2.cvtColor(img_base_color, cv2.COLOR_BGR2GRAY)
    warp_gray = cv2.cvtColor(img_to_warp_color, cv2.COLOR_BGR2GRAY)

    hB, wB = base_gray.shape[:2]
    hW, wW = warp_gray.shape[:2]

    # Esquinas del plano de la imagen a proyectar (src)
    corners = np.float32([[0, 0], [0, hW], [wW, hW], [wW, 0]]).reshape(-1, 1, 2)
    # Transformarlas con H para conocer el bounding box en el destino
    transformed = cv2.perspectiveTransform(corners, H)

    # Ajuste de límites y traslación para evitar índices negativos
    [xmin, ymin] = np.int32(np.round(transformed.min(axis=0).ravel() - 0.5))
    [xmax, ymax] = np.int32(np.round(transformed.max(axis=0).ravel() + 0.5))

    tx = max(0, -xmin)
    ty = max(0, -ymin)

    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

    H_final = T @ H

    out_w = int(xmax - xmin + tx)
    out_h = int(ymax - ymin + ty)

    # Proyectar la imagen warp al canvas (gris por simplicidad de máscaras)
    mosaic_gray = cv2.warpPerspective(warp_gray, H_final, (out_w, out_h))

    # ROI donde colocar la imagen base dentro del canvas
    y0, y1 = ty, ty + hB
    x0, x1 = tx, tx + wB

    # Recortar si la ROI excede los límites del canvas
    roi = mosaic_gray[y0:y1, x0:x1]
    h_clip = min(roi.shape[0], base_gray.shape[0])
    w_clip = min(roi.shape[1], base_gray.shape[1])
    roi = roi[:h_clip, :w_clip]
    base_clip = base_gray[:h_clip, :w_clip]

    # Máscara 1: dónde la base tiene contenido (no negro)
    mask_base = base_clip != 0
    # Máscara 2: dónde el canvas aún está vacío (negro)
    mask_empty = roi == 0
    # Pegamos la base SOLO donde el canvas está vacío
    mask_final = mask_base & mask_empty
    roi[mask_final] = base_clip[mask_final]

    # Reinyectar la ROI modificada al canvas
    mosaic_gray[y0 : y0 + h_clip, x0 : x0 + w_clip] = roi

    # Devolver en BGR (para consistencia con el resto del pipeline)
    return cv2.cvtColor(mosaic_gray, cv2.COLOR_GRAY2BGR)
