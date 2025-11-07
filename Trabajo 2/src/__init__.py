import sys

sys.path.append("..")  # subir un nivel desde notebooks/

from src.feature_detection import detectar_y_describir
from src.matching import emparejar_y_filtrar
from src.registration import estimar_homografia
from src.stitching import unificar_dos_imagenes
