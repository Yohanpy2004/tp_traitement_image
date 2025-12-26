# src/processing/spatial_filters.py
import cv2
import numpy as np

def apply_average_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Applique un filtre moyenneur à une image.
    Nous construisons le noyau manuellement et utilisons la convolution optimisée d'OpenCV.
    
    Args:
        image (np.ndarray): L'image d'entrée.
        kernel_size (int): La taille du noyau (doit être un entier impair).
        
    Returns:
        np.ndarray: L'image filtrée.
    """
    # Assurer que la taille du noyau est impaire
    if kernel_size % 2 == 0:
        raise ValueError("La taille du noyau doit être impaire.")
        
    # 1. Création du noyau (masque)
    # Le noyau est une matrice de taille kernel_size x kernel_size
    # dont tous les éléments sont égaux à 1 / (nombre total d'éléments).
    # Cela garantit que la somme des coefficients du noyau est égale à 1.
    kernel_value = 1.0 / (kernel_size * kernel_size)
    kernel = np.full((kernel_size, kernel_size), kernel_value, dtype=np.float32)
    
    # 2. Application de la convolution
    # cv2.filter2D est la fonction de convolution 2D.
    # L'argument -1 pour 'ddepth' signifie que l'image de sortie aura la même
    # profondeur (type de données) que l'image d'entrée.
    # On spécifie le traitement des bords avec 'borderType'
    filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    
    return filtered_image

# Dans src/processing/spatial_filters.py




# ... (la fonction apply_average_filter reste là) ...

def apply_gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Applique un filtre gaussien. Le noyau est généré manuellement.
    
    Args:
        image (np.ndarray): L'image d'entrée.
        kernel_size (int): La taille du noyau (doit être impair).
        sigma (float): L'écart-type de la gaussienne.
        
    Returns:
        np.ndarray: L'image filtrée.
    """
    if kernel_size % 2 == 0:
        raise ValueError("La taille du noyau doit être impaire.")

    # Création du noyau Gaussien 2D
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    
    # Normalisation du noyau pour que sa somme soit égale à 1
    kernel = kernel / np.sum(kernel)
    
    # Application de la convolution
    filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REPLICATE)
    
    return filtered_image





# Dans src/processing/spatial_filters.py

# ... (les fonctions précédentes restent là) ...

def apply_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Applique un filtre médian, qui est un filtre non-linéaire.
    Efficace pour supprimer le bruit de type "poivre et sel".
    
    Args:
        image (np.ndarray): L'image d'entrée.
        kernel_size (int): La taille du voisinage (doit être impair).
        
    Returns:
        np.ndarray: L'image filtrée.
    """
    if kernel_size % 2 == 0:
        raise ValueError("La taille du noyau doit être impaire.")
        
    # OpenCV fournit une implémentation C++ optimisée de ce filtre non-linéaire.
    filtered_image = cv2.medianBlur(image, kernel_size)
    
    return filtered_image


def apply_sharpen_filter(image: np.ndarray) -> np.ndarray:
    """
    Applique un filtre de rehaussement de netteté simple.
    """
    # Ce noyau accentue la différence entre le pixel central et ses voisins.
    # La somme des coefficients est 1, donc la luminosité est préservée.
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
    
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)