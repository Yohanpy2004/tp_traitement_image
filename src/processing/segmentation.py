# src/processing/segmentation.py

import cv2
import numpy as np
from src.processing.point_operations import convert_to_grayscale

def simple_thresholding(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Applique un seuillage binaire simple.
    Les pixels > threshold deviennent blancs (255), les autres noirs (0).
    
    Args:
        image (np.ndarray): Image d'entrée.
        threshold (int): La valeur de seuil (0-255).
        
    Returns:
        np.ndarray: L'image binaire.
    """
    gray_image = convert_to_grayscale(image)
    
    # NumPy est extrêmement efficace pour ce type de comparaison booléenne.
    # On crée une image de la même taille, remplie de zéros (noir).
    binary_image = np.zeros_like(gray_image)
    
    # Partout où la condition (gray_image > threshold) est vraie, on met la valeur à 255.
    binary_image[gray_image > threshold] = 255
    
    return binary_image

def otsu_thresholding(image: np.ndarray) -> np.ndarray:
    """
    Applique un seuillage automatique en utilisant la méthode d'Otsu.
    Otsu trouve le seuil optimal qui minimise la variance intra-classe.
    
    Args:
        image (np.ndarray): Image d'entrée.
        
    Returns:
        np.ndarray: L'image binaire.
    """
    gray_image = convert_to_grayscale(image)
    
    # La fonction d'OpenCV est une implémentation optimisée de l'algorithme d'Otsu.
    # Elle retourne le seuil trouvé et l'image binarisée.
    # cv2.THRESH_BINARY indique que les pixels > seuil -> 255, sinon -> 0.
    # cv2.THRESH_OTSU active la méthode d'Otsu pour calculer le seuil.
    optimal_threshold, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # On pourrait afficher le seuil trouvé si on le voulait (ex: st.info(f"Seuil d'Otsu trouvé : {optimal_threshold}") )
    
    return binary_image

def adaptive_thresholding(image: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """
    Applique un seuillage adaptatif. Le seuil est calculé localement pour chaque pixel.
    Très efficace pour les images avec des conditions d'éclairage variables.
    
    Args:
        image (np.ndarray): Image d'entrée.
        block_size (int): Taille du voisinage pour calculer le seuil (doit être impair).
        C (int): Constante soustraite de la moyenne locale.
        
    Returns:
        np.ndarray: L'image binaire.
    """
    gray_image = convert_to_grayscale(image)
    
    # Assurer que la taille du bloc est impaire
    if block_size % 2 == 0:
        block_size += 1
        
    # cv2.ADAPTIVE_THRESH_MEAN_C: le seuil est la moyenne du voisinage moins C.
    # cv2.THRESH_BINARY: même règle que pour le seuillage simple.
    binary_image = cv2.adaptiveThreshold(
        src=gray_image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=C
    )
    
    return binary_image