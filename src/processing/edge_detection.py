# src/processing/edge_detection.py
import cv2
import numpy as np
# On importe la fonction de conversion en niveaux de gris qu'on a déjà faite !
from src.processing.point_operations import convert_to_grayscale

def apply_sobel(image: np.ndarray) -> np.ndarray:
    """
    Applique l'opérateur de Sobel pour détecter les contours dans une image.
    
    Args:
        image (np.ndarray): L'image d'entrée (peut être en couleur ou niveaux de gris).
        
    Returns:
        np.ndarray: Une image en niveaux de gris montrant la magnitude du gradient.
    """
    # Étape 1: S'assurer que l'image est en niveaux de gris
    gray_image = convert_to_grayscale(image)
    
    # Étape 2: Définir les noyaux de Sobel Gx et Gy manuellement
    # Noyau pour le gradient horizontal (détecte les contours verticaux)
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # Noyau pour le gradient vertical (détecte les contours horizontaux)
    kernel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)
    
    # Étape 3: Appliquer les convolutions
    # On utilise un ddepth plus grand (cv2.CV_64F) pour éviter de perdre de l'information
    # avec les valeurs négatives du gradient, puis on remet à l'échelle.
    grad_x = cv2.filter2D(src=gray_image, ddepth=cv2.CV_64F, kernel=kernel_x)
    grad_y = cv2.filter2D(src=gray_image, ddepth=cv2.CV_64F, kernel=kernel_y)
    
    # Étape 4: Calculer la magnitude du gradient
    # G = sqrt(Gx^2 + Gy^2)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Étape 5: Normaliser le résultat pour l'affichage
    # On remet l'échelle des valeurs du gradient dans l'intervalle 0-255.
    # cv2.convertScaleAbs calcule la valeur absolue puis convertit en uint8.
    sobel_output = cv2.convertScaleAbs(gradient_magnitude)
    
    return sobel_output

# src/processing/edge_detection.py (Version mise à jour)
import cv2
import numpy as np
from src.processing.point_operations import convert_to_grayscale

# La fonction apply_sobel reste ici, inchangée...
# def apply_sobel(image: np.ndarray) -> np.ndarray:
#     # ... (code existant) ...
#     gray_image = convert_to_grayscale(image)
#     kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
#     kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
#     grad_x = cv2.filter2D(src=gray_image, ddepth=cv2.CV_64F, kernel=kernel_x)
#     grad_y = cv2.filter2D(src=gray_image, ddepth=cv2.CV_64F, kernel=kernel_y)
#     gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
#     sobel_output = cv2.convertScaleAbs(gradient_magnitude)
#     return sobel_output

# --- NOUVELLES FONCTIONS ---

def apply_prewitt(image: np.ndarray) -> np.ndarray:
    """Applique l'opérateur de Prewitt."""
    gray_image = convert_to_grayscale(image)
    
    kernel_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)
    
    grad_x = cv2.filter2D(src=gray_image, ddepth=cv2.CV_64F, kernel=kernel_x)
    grad_y = cv2.filter2D(src=gray_image, ddepth=cv2.CV_64F, kernel=kernel_y)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    prewitt_output = cv2.convertScaleAbs(gradient_magnitude)
    return prewitt_output

def apply_laplacian(image: np.ndarray) -> np.ndarray:
    """Applique l'opérateur Laplacien."""
    gray_image = convert_to_grayscale(image)
    
    # Noyau Laplacien standard (4-connexité)
    kernel = np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float32)
    
    laplacian_output = cv2.filter2D(src=gray_image, ddepth=cv2.CV_64F, kernel=kernel)
    laplacian_output = cv2.convertScaleAbs(laplacian_output)
    return laplacian_output



def apply_roberts(image: np.ndarray) -> np.ndarray:
    """Applique l'opérateur de Roberts."""
    gray_image = convert_to_grayscale(image)
    # Noyaux 2x2
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    grad_x = cv2.filter2D(src=gray_image, ddepth=cv2.CV_64F, kernel=kernel_x)
    grad_y = cv2.filter2D(src=gray_image, ddepth=cv2.CV_64F, kernel=kernel_y)
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return cv2.convertScaleAbs(gradient_magnitude)

def apply_canny(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Applique l'algorithme de Canny pour la détection de contours.
    C'est un algorithme multi-étapes optimisé dans OpenCV.
    
    Args:
        image (np.ndarray): Image d'entrée.
        low_threshold (int): Premier seuil pour l'hystérésis.
        high_threshold (int): Second seuil pour l'hystérésis.
        
    Returns:
        np.ndarray: Image binaire des contours.
    """
    gray_image = convert_to_grayscale(image)
    # cv2.Canny est une fonction hautement optimisée qui effectue le lissage,
    # le calcul du gradient, la suppression des non-maxima et le seuillage par hystérésis.
    return cv2.Canny(gray_image, low_threshold, high_threshold)