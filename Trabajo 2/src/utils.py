def normalizar_homografia(H):
    """
    Normaliza una matriz de Homografía H dividiendo por H[2, 2] para que H[2, 2] sea 1.
    Esto es esencial para comparar Homografías.
    """
    if H is not None:
        # Verifica que el elemento de escala inferior derecho no sea cero
        if H[2, 2] != 0:
            return H / H[2, 2]
        else:
            # En caso de matriz degenerada, se devuelve tal cual (o se maneja el error)
            return H
    return None
