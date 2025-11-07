"""
registration.py
---------------
Estimación de homografías entre pares de imágenes a partir de correspondencias
de puntos (matches) y cálculo de métricas básicas de calidad.

Funciones:
    - normalizar_homografia(H)
    - estimar_homografia(kp_src, kp_dst, matches, ransac_thresh=5.0)

Nota: se usa cv2.findHomography con RANSAC para filtrar outliers.
"""

import numpy as np
import cv2


def normalizar_homografia(H):
    """
    Normaliza una homografía H para que H[2,2] = 1. Esto facilita comparaciones.
    """
    if H is not None and H.shape == (3, 3) and H[2, 2] != 0:
        return H / H[2, 2]
    return H


def estimar_homografia(kp_src, kp_dst, matches, ransac_thresh=5.0):
    """
    Estima la homografía que transforma puntos de la imagen 'src' a la imagen 'dst'.

    Parámetros
    ----------
    kp_src : list[cv2.KeyPoint]
        Keypoints de la imagen fuente (puntos que se proyectan).
    kp_dst : list[cv2.KeyPoint]
        Keypoints de la imagen destino/base (puntos "reales" en el mosaico).
    matches : list[cv2.DMatch]
        Lista de correspondencias filtradas (output de emparejar_y_filtrar).
        IMPORTANTE: matches[i].queryIdx indexa kp_src; .trainIdx indexa kp_dst.
    ransac_thresh : float
        Umbral de reproyección para RANSAC en píxeles (típico 3–10 px).

    Retorna
    -------
    H : np.ndarray | None
        Homografía 3x3 normalizada (o None si falló).
    inliers_mask : np.ndarray | None
        Máscara booleana (N,) indicando qué matches son inliers.
    metrics : dict
        Métricas básicas: número de matches, inliers y ratio de inliers.

    Notas
    -----
    - Requiere al menos 4 correspondencias válidas para estimar una homografía.
    """
    if matches is None or len(matches) < 4:
        return None, None, {"num_matches": 0, "num_inliers": 0, "inlier_ratio": 0.0}

    # Construcción de arreglos con coordenadas (x, y) en el orden correcto.
    pts_src = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_dst = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimar H con RANSAC (devuelve la máscara de inliers)
    H, inliers_mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, ransac_thresh)

    if H is None or inliers_mask is None:
        return (
            None,
            None,
            {"num_matches": len(matches), "num_inliers": 0, "inlier_ratio": 0.0},
        )

    inliers_mask = inliers_mask.ravel().astype(bool)
    metrics = {
        "num_matches": int(len(matches)),
        "num_inliers": int(inliers_mask.sum()),
        "inlier_ratio": float(inliers_mask.mean()),
    }

    return normalizar_homografia(H), inliers_mask, metrics
