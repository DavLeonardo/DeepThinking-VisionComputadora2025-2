"""
feature_detection.py
--------------------
Rutinas para detectar y describir características locales (keypoints) en imágenes,
utilizando detectores/descriptores clásicos como SIFT, ORB y AKAZE.

Este módulo expone una única función de alto nivel:
    - detectar_y_describir(img_gray, detector_tipo="SIFT", **kwargs)

La idea es desacoplar la elección del detector del resto del pipeline para poder
experimentar fácilmente (cambiar SIFT por ORB, etc.).

Requisitos:
- OpenCV con contrib para SIFT/AKAZE: opencv-contrib-python
"""

import cv2


def detectar_y_describir(img_gray, detector_tipo="SIFT", **kwargs):
    """
    Detecta puntos clave (keypoints) y calcula descriptores para una imagen
    en escala de grises usando el detector especificado.

    Parámetros
    ----------
    img_gray : np.ndarray
        Imagen de entrada en escala de grises (dtype uint8).
    detector_tipo : str, opcional
        Tipo de detector/descriptor a utilizar. Opciones:
        "SIFT" (requiere opencv-contrib), "ORB", "AKAZE".
        Por defecto, "SIFT".
    **kwargs :
        Parámetros específicos para algunos detectores (ej., nfeatures para ORB).

    Retorna
    -------
    kp : list[cv2.KeyPoint]
        Lista de keypoints detectados por el detector.
    des : np.ndarray | None
        Matriz de descriptores (cada fila corresponde a un keypoint). Puede ser None
        si no se detectan suficientes puntos.

    Notas
    -----
    - Si el detector solicitado no existe, se caerá a SIFT por defecto.
    - Para SIFT/AKAZE se retornan descriptores tipo float32; para ORB, binarios.
    """
    if img_gray is None:
        raise ValueError(
            "img_gray es None. Asegúrate de convertir a gris con cv2.cvtColor."
        )

    # Selección/creación del detector según el tipo solicitado.
    # Se permiten kwargs para facilitar la experimentación (p.ej., nfeatures para ORB).
    detector_tipo = (detector_tipo or "SIFT").upper().strip()
    if detector_tipo == "SIFT":
        # SIFT (robusto, pero requiere opencv-contrib)
        detector = cv2.SIFT_create()
    elif detector_tipo == "ORB":
        # ORB (rápido, binario). Permite nfeatures.
        nfeatures = kwargs.get("nfeatures", 1000)
        detector = cv2.ORB_create(nfeatures=nfeatures)
    elif detector_tipo == "AKAZE":
        detector = cv2.AKAZE_create()
    else:
        # Fallback a SIFT para no romper el flujo si hay un typo
        detector = cv2.SIFT_create()

    # Detección y descripción (una sola pasada)
    kp, des = detector.detectAndCompute(img_gray, None)

    return kp, des
