"""
matching.py
-----------
Emparejamiento de descriptores entre pares de imágenes y filtrado robusto usando
el test de razón de Lowe (Lowe's ratio test).

Exposición principal:
    - emparejar_y_filtrar(des1, des2, detector_tipo="SIFT", ratio_umbral=0.75)

Para SIFT/AKAZE (descriptores float), se usa FLANN (rápido).
Para ORB (descriptores binarios), se usa BFMatcher con NORM_HAMMING.
"""

import cv2


def emparejar_y_filtrar(des1, des2, detector_tipo="SIFT", ratio_umbral=0.75):
    """
    Empareja descriptores entre dos imágenes y filtra correspondencias
    usando el test de razón (Lowe).

    Parámetros
    ----------
    des1, des2 : np.ndarray
        Descriptores de la imagen1 e imagen2, respectivamente.
        OJO: El orden importa luego al construir homografías (src -> dst).
    detector_tipo : str
        "SIFT" | "AKAZE" => FLANN; "ORB" => BFMatcher(NORM_HAMMING).
    ratio_umbral : float
        Umbral para el test de razón. 0.75 es un valor típico.

    Retorna
    -------
    buenos : list[cv2.DMatch]
        Lista de emparejamientos filtrados (correspondencias plausibles).
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        # Si no hay suficientes descriptores, no se puede aplicar KNN (k=2)
        return []

    detector_tipo = (detector_tipo or "SIFT").upper().strip()

    # Selección del "matcher" según el tipo de descriptor.
    if detector_tipo in ("SIFT", "AKAZE"):
        # FLANN para descriptores de punto flotante
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # más checks => más robusto, más lento
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    elif detector_tipo == "ORB":
        # BF con distancia Hamming para descriptores binarios
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        # fallback razonable
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # KNN match con k=2 para poder aplicar Lowe's ratio test
    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        # Si hay incompatibilidad de tipos/formatos esto captura el error
        return []

    # Lowe's ratio test: conserva matches con m.distance < ratio * n.distance
    buenos = []
    for m, n in matches:
        if m.distance < ratio_umbral * n.distance:
            buenos.append(m)

    return buenos
