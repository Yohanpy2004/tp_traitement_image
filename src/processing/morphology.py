# src/processing/morphology.py

import cv2
import numpy as np
from .point_operations import convert_to_grayscale

def get_structuring_element(shape: str, size: int) -> np.ndarray:
    """
    Crée un élément structurant (noyau) pour les opérations morphologiques.
    
    Args:
        shape (str): La forme de l'élément ('rectangle', 'ellipse', 'croix').
        size (int): La taille de l'élément.
        
    Returns:
        np.ndarray: Le noyau pour les opérations morphologiques.
    """
    if shape == 'rectangle':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif shape == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif shape == 'croix':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    else:
        raise ValueError("Forme d'élément structurant non reconnue. Choisissez parmi : 'rectangle', 'ellipse', 'croix'.")

def erosion(image: np.ndarray, kernel_shape: str = 'rectangle', kernel_size: int = 3) -> np.ndarray:
    """
    Applique une érosion morphologique.
    Rétrécit les régions blanches. Utile pour supprimer le bruit blanc et séparer les objets.
    
    Args:
        image (np.ndarray): Image d'entrée, idéalement binaire.
        kernel_shape (str): Forme de l'élément structurant.
        kernel_size (int): Taille de l'élément structurant.
        
    Returns:
        np.ndarray: Image érodée.
    """
    # L'érosion fonctionne mieux sur des images binaires (0 ou 255)
    # On s'assure que l'image d'entrée est bien binaire
    if len(np.unique(image)) > 2:
        _, binary_image = cv2.threshold(convert_to_grayscale(image), 127, 255, cv2.THRESH_BINARY)
    else:
        binary_image = image

    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.erode(binary_image, kernel, iterations=1)

def dilation(image: np.ndarray, kernel_shape: str = 'rectangle', kernel_size: int = 3) -> np.ndarray:
    """
    Applique une dilatation morphologique.
    Épaissit les régions blanches. Utile pour combler les trous et connecter des composants.
    
    Args:
        image (np.ndarray): Image d'entrée, idéalement binaire.
        kernel_shape (str): Forme de l'élément structurant.
        kernel_size (int): Taille de l'élément structurant.
        
    Returns:
        np.ndarray: Image dilatée.
    """
    if len(np.unique(image)) > 2:
        _, binary_image = cv2.threshold(convert_to_grayscale(image), 127, 255, cv2.THRESH_BINARY)
    else:
        binary_image = image
        
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.dilate(binary_image, kernel, iterations=1)

def opening(image: np.ndarray, kernel_shape: str = 'rectangle', kernel_size: int = 3) -> np.ndarray:
    """
    Applique une ouverture morphologique (érosion suivie d'une dilatation).
    Très efficace pour supprimer le bruit "poivre" (petits points blancs) sans
    affecter la taille globale des objets principaux.
    
    Args:
        image (np.ndarray): Image d'entrée, idéalement binaire.
        kernel_shape (str): Forme de l'élément structurant.
        kernel_size (int): Taille de l'élément structurant.
        
    Returns:
        np.ndarray: Image après ouverture.
    """
    if len(np.unique(image)) > 2:
        _, binary_image = cv2.threshold(convert_to_grayscale(image), 127, 255, cv2.THRESH_BINARY)
    else:
        binary_image = image
        
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

def closing(image: np.ndarray, kernel_shape: str = 'rectangle', kernel_size: int = 3) -> np.ndarray:
    """
    Applique une fermeture morphologique (dilatation suivie d'une érosion).
    Très efficace pour combler les petits trous noirs dans les objets ("sel") sans
    changer la forme générale.
    
    Args:
        image (np.ndarray): Image d'entrée, idéalement binaire.
        kernel_shape (str): Forme de l'élément structurant.
        kernel_size (int): Taille de l'élément structurant.
        
    Returns:
        np.ndarray: Image après fermeture.
    """
    if len(np.unique(image)) > 2:
        _, binary_image = cv2.threshold(convert_to_grayscale(image), 127, 255, cv2.THRESH_BINARY)
    else:
        binary_image = image
        
    kernel = get_structuring_element(kernel_shape, kernel_size)
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)