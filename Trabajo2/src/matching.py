import cv2
import numpy as np
from utils import *
from matplotlib import pyplot as plt
import os

def detectar_y_describir(imagen, detector_tipo="SIFT"):
    """
    Detecta keypoints y calcula descriptores para una imagen usando el algoritmo especificado.
    
    Parámetros:
    - imagen: Imagen de entrada (preferiblemente escala de grises).
    - detector_tipo: String que especifica el algoritmo ('SIFT', 'ORB', 'AKAZE').
    
    Retorna:
    - kp: Lista de KeyPoints de OpenCV.
    - des: Array de descriptores (NumPy).
    """
    
    if detector_tipo == "SIFT":
        detector = cv2.SIFT_create()
    elif detector_tipo == "ORB":
        detector = cv2.ORB_create(nfeatures=1000) # Se puede ajustar nfeatures
    elif detector_tipo == "AKAZE":
        detector = cv2.AKAZE_create()
    else:
        print(f"Advertencia: Detector '{detector_tipo}' no reconocido. Usando SIFT por defecto.")
        detector = cv2.SIFT_create()

    # Detectar Keypoints y calcular descriptores
    kp, des = detector.detectAndCompute(imagen, None)
    
    return kp, des

def emparejar_y_filtrar(des1, des2, detector_tipo="SIFT", ratio_umbral=0.75):
    """
    Empareja descriptores entre dos imágenes y aplica el filtro de ratio de Lowe.
    
    Parámetros:
    - des1, des2: Descriptores de las dos imágenes.
    - detector_tipo: Para seleccionar el tipo de Matcher (BFMatcher o FlannBasedMatcher).
    - ratio_umbral: Umbral para el ratio test (0.75 es un valor común).
    
    Retorna:
    - buenos_matches: Lista de objetos DMatch filtrados.
    """
    
    # 1. Seleccionar Matcher
    if detector_tipo in ["SIFT", "AKAZE"]:
        # Para descriptores flotantes, Flann es más rápido
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # o 50
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
    elif detector_tipo == "ORB":
        # Para descriptores binarios (ORB), se usa Brute Force con NORM_HAMMING
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    else:
        # Por defecto, se usa FlannBasedMatcher si los descriptores son flotantes
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # 2. Aplicar KNN Match
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []

    try:
        matches = matcher.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        print(f"Error en knnMatch: {e}")
        return []
    
    # 3. Aplicar Filtro de Ratio de Lowe
    buenos_matches = []
    for m, n in matches:
        if m.distance < ratio_umbral * n.distance:
            buenos_matches.append(m)
            
    return buenos_matches

def estimate_homography_and_metrics(kp1, kp2, matches, H_gt=None, ransac_thresh=5.0):
    """
    Estima la homografía (matriz 3x3) y calcula métricas, incluyendo una prueba 
    de inversión para resolver problemas de convención H_gt vs H_est.
    """
    if len(matches) < 4:
        return None, None, {"num_matches": len(matches), "num_inliers": 0, "inlier_ratio": 0.0}, None

    # pts1 (Source) -> pts2 (Destination)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimar Homografía H_est usando RANSAC
    H_est, inliers = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    
    if H_est is None:
        return None, None, {"num_matches": len(matches), "num_inliers": 0, "inlier_ratio": 0.0}, None

    # Filtrar inliers
    inliers = inliers.ravel().astype(bool)
    corrected_matches = [m for m, inl in zip(matches, inliers) if inl]

    metrics = {
        "num_matches": len(matches),
        "num_inliers": np.sum(inliers),
        "inlier_ratio": np.sum(inliers) / len(matches)
    }

    # CÁLCULO DE RMSE DE REPROYECCIÓN
    pts1_inliers_h = cv2.convertPointsToHomogeneous(pts1[inliers]).reshape(-1, 3)
    proj = (H_est @ pts1_inliers_h.T).T 
    proj /= proj[:, 2:3]
    proj = proj[:, :2]

    error = np.linalg.norm(proj - pts2[inliers].reshape(-1, 2), axis=1)
    metrics["rmse_reprojection"] = np.sqrt(np.mean(np.square(error)))


    # CÁLCULO DE MÉTRICAS DE ERROR DE HOMOGRAFÍA (Ground Truth)
    if H_gt is not None:
        
        I = np.eye(3)
        
        # 1. Normalización de Ambas Homografías (usando la función de la Celda 3)
        H_gt_norm = normalizar_homografia(H_gt)
        H_est_norm = normalizar_homografia(H_est)
        
        # 2. PRUEBA DE CONVENCIÓN DE ERROR
        # Prueba A: E = H_gt_inv * H_est (La fórmula teórica)
        try:
            E_directa = np.linalg.inv(H_gt_norm) @ H_est_norm
            error_directo = np.linalg.norm(E_directa - I)
        except np.linalg.LinAlgError:
            error_directo = 9999.0 

        # Prueba B: E_alt = H_gt * H_est_inv (Prueba inversa para diagnosticar convención)
        try:
            E_alterna = H_gt_norm @ np.linalg.inv(H_est_norm)
            error_alterno = np.linalg.norm(E_alterna - I)
        except np.linalg.LinAlgError:
            error_alterno = 9999.0

        # 3. Selección del Resultado Más Preciso
        if error_directo < error_alterno:
            metrics["homography_error"] = error_directo
            E_final = E_directa
            metrics["error_convencion"] = "Directa (H_gt_inv * H_est)" 
        else:
            metrics["homography_error"] = error_alterno
            E_final = E_alterna
            metrics["error_convencion"] = "Alterna (H_gt * H_est_inv)"
        
        # 4. Cálculo del Error Angular Estable (SVD sobre la matriz de error E_final)
        E_rot = E_final[:2, :2] 
        U, _, Vt = np.linalg.svd(E_rot)
        R_E = U @ Vt
        residual_angle_rad = np.arctan2(R_E[1, 0], R_E[0, 0])
        metrics["angular_error_deg"] = np.degrees(np.abs(residual_angle_rad))

    return H_est, inliers, metrics, corrected_matches

def draw_matches_and_save(img1, kp1, img2, kp2, matches, title,base_dir, filename):
    """
    Dibuja los matches entre dos imágenes y los guarda como figura.
    """
    # Dibujar matches. Se recomienda dibujar solo una fracción si hay muchos
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    
    # Matplotlib para visualización en el notebook
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

    cv2.imwrite(os.path.join(base_dir, filename), img_matches)
    print(f"Figura guardada en: {os.path.join(base_dir, filename)}")
    
    
def unificar_dos_imagenes(img_base_color, img_to_warp_color, H):
    """
    Unifica dos imágenes dadas una Homografía H. Utiliza una máscara doble 
    para manejar correctamente los fondos negros en mosaicos de varios pasos.
    """
    
    # 1. Preparación y Detección de Límites
    img_base = cv2.cvtColor(img_base_color, cv2.COLOR_BGR2GRAY)
    img_to_warp = cv2.cvtColor(img_to_warp_color, cv2.COLOR_BGR2GRAY)
    
    alto_base, ancho_base = img_base.shape[:2]
    alto_warp, ancho_warp = img_to_warp.shape[:2]

    # Transformar esquinas y calcular desplazamiento (Esta parte es correcta)
    corners = np.float32([[0, 0], [0, alto_warp], [ancho_warp, alto_warp], [ancho_warp, 0]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)

    # Nota: Usamos round() para minimizar errores de float antes de int()
    [xmin, ymin] = np.int32(np.round(transformed_corners.min(axis=0).ravel() - 0.5))
    [xmax, ymax] = np.int32(np.round(transformed_corners.max(axis=0).ravel() + 0.5))

    tx_desplazamiento = max(0, -xmin)
    ty_desplazamiento = max(0, -ymin)

    T_translate = np.array([[1, 0, tx_desplazamiento], [0, 1, ty_desplazamiento], [0, 0, 1]], dtype=np.float32)
    H_final = T_translate @ H

    output_width = int(xmax - xmin + tx_desplazamiento)
    output_height = int(ymax - ymin + ty_desplazamiento)

    # 2. Proyección de la Imagen Deformada
    # Crear mosaico y proyectar imagen deformada
    img_unificada = np.zeros((output_height, output_width), dtype=img_base.dtype)
    img_unificada = cv2.warpPerspective(img_to_warp, H_final, (output_width, output_height), dst=img_unificada, borderMode=cv2.BORDER_TRANSPARENT)

    # 3. Colocar la Imagen Base (A+B) con DOBLE MÁSCARA (La Corrección Final)
    
    ROI_y_start = ty_desplazamiento; ROI_y_end = ty_desplazamiento + alto_base
    ROI_x_start = tx_desplazamiento; ROI_x_end = tx_desplazamiento + ancho_base
    
    # Obtener la ROI del mosaico
    roi_unificada = img_unificada[ROI_y_start:ROI_y_end, ROI_x_start:ROI_x_end]
    
    # --- Verificación de Dimensiones Crítica (Diagnóstico) ---
    if roi_unificada.shape != img_base.shape:
        print(f"!!! DIAGNÓSTICO: Mismatch Dimensional entre ROI ({roi_unificada.shape}) e img_base ({img_base.shape}) !!!")
        print("Este mismatch es la causa del IndexError. Ajuste los límites de la ROI para que coincidan.")
        # Ajustamos el slicing para que coincida con la imagen base (se recorta si es necesario)
        h_clip = min(roi_unificada.shape[0], img_base.shape[0])
        w_clip = min(roi_unificada.shape[1], img_base.shape[1])
        roi_unificada = roi_unificada[:h_clip, :w_clip]
        img_base_clipped = img_base[:h_clip, :w_clip]
    else:
        img_base_clipped = img_base

    # 1. Máscara de Contenido Base: TRUE donde la imagen base (A+B) tiene píxeles (no-negro)
    mascara_contenido_base = (img_base_clipped != 0)
    
    # 2. Máscara de Vacío del Canvas: TRUE donde el canvas (C proyectada) es negro
    mascara_vacio_canvas = (roi_unificada == 0)
    
    # 3. Máscara Final: Intersección (pegar el contenido de A+B SOLO donde el canvas está vacío)
    mascara_final = mascara_contenido_base & mascara_vacio_canvas
    
    # 4. Aplicar la imagen base SÓLO en las áreas vacías.
    roi_unificada[mascara_final] = img_base_clipped[mascara_final] 
    
    # 5. Volver a asignar la ROI modificada al mosaico final
    # Usamos las coordenadas de la ROI original (sin clip) para la asignación final
    img_unificada[ROI_y_start:ROI_y_end, ROI_x_start:ROI_x_end] = roi_unificada

    # Retornar la imagen en color
    return cv2.cvtColor(img_unificada, cv2.COLOR_GRAY2BGR)