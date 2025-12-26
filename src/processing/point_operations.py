# src/processing/point_operations.py
import cv2
import numpy as np

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convertit une image couleur (BGR) en niveaux de gris en utilisant la formule de luminance.
    L'implémentation est vectorisée avec NumPy pour une efficacité maximale.
    
    Args:
        image (np.ndarray): L'image d'entrée au format BGR (comme chargée par OpenCV).
                           Doit être un tableau 3D.
        
    Returns:
        np.ndarray: L'image en niveaux de gris (tableau 2D).
    """
    # On vérifie d'abord si l'image n'est pas déjà en niveaux de gris
    if len(image.shape) == 2:
        return image.copy() # C'est déjà fait, on retourne une copie
    
    # OpenCV charge les images en BGR, donc les canaux sont (Bleu, Vert, Rouge)
    # On extrait les canaux
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    # Application de la formule de luminance (standard ITU-R BT.601)
    # L'opération est faite sur toute la matrice d'un coup, c'est très rapide.
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Le résultat est un float, on le reconvertit en entier non signé 8 bits
    # (le format standard pour les images)
    gray_image = gray_image.astype(np.uint8)
    
    return gray_image

def invert_image(image: np.ndarray) -> np.ndarray:
    """
    Inverse les couleurs d'une image (négatif).
    Cette opération est aussi vectorisée.
    
    Args:
        image (np.ndarray): L'image d'entrée (couleur ou niveaux de gris).
        
    Returns:
        np.ndarray: L'image inversée.
    """
    # L'opération est très simple : 255 - valeur du pixel.
    # NumPy l'applique à chaque pixel de l'image simultanément.
    inverted_image = 255 - image
    return inverted_image

# Dans src/processing/point_operations.py

# ... (les fonctions convert_to_grayscale et invert_image restent là) ...

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Applique l'égalisation d'histogramme à une image en niveaux de gris.
    Implémentation manuelle utilisant NumPy.
    
    Args:
        image (np.ndarray): L'image d'entrée (sera convertie en niveaux de gris si nécessaire).
        
    Returns:
        np.ndarray: L'image égalisée.
    """
    # L'égalisation ne fonctionne que sur les images en niveaux de gris.
    if len(image.shape) > 2:
        gray_image = convert_to_grayscale(image)
    else:
        gray_image = image.copy()

    # Étape 1: Calculer l'histogramme
    # np.histogram compte les occurrences de chaque intensité (0-255)
    # Le 'bins' est 256 pour [0, 1, ..., 255]. Le 'range' est [0, 256] (borne supérieure exclusive).
    hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    
    # Étape 2: Calculer l'histogramme cumulé (CDF)
    cdf = hist.cumsum()
    
    # Étape 3: Normaliser la CDF pour créer la table de correspondance (LUT)
    # On masque les valeurs nulles de la cdf pour éviter les divisions par zéro
    # et pour que ces niveaux de gris restent à 0.
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # Formule de normalisation
    num_pixels = gray_image.shape[0] * gray_image.shape[1]
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (num_pixels - cdf_m.min())
    
    # Remplir les valeurs masquées avec 0
    cdf_normalized = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # Étape 4: Appliquer la LUT à l'image
    equalized_image = cdf_normalized[gray_image]
    
    return equalized_image



def adjust_brightness_contrast(image: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """
    Ajuste la luminosité et le contraste de l'image (transformation linéaire g = alpha*f + beta).
    
    Args:
        image (np.ndarray): Image d'entrée.
        alpha (float): Contrôle du contraste ( > 1 pour augmenter).
        beta (int): Contrôle de la luminosité ( > 0 pour éclaircir).
        
    Returns:
        np.ndarray: Image ajustée.
    """
    # Utilise la fonction optimisée d'OpenCV qui applique la formule et gère le clipping (0-255)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def contrast_stretching(image: np.ndarray, min_out: int = 0, max_out: int = 255) -> np.ndarray:
    """
    Applique un étirement de contraste en mappant les valeurs d'entrée sur une nouvelle plage.
    
    Args:
        image (np.ndarray): Image d'entrée (sera convertie en niveaux de gris).
        min_out (int): La nouvelle valeur minimale de l'histogramme (0-255).
        max_out (int): La nouvelle valeur maximale de l'histogramme (0-255).
        
    Returns:
        np.ndarray: Image avec contraste étiré.
    """
    if len(image.shape) > 2:
        img_in = convert_to_grayscale(image)
    else:
        img_in = image.copy()
        
    min_in = np.min(img_in)
    max_in = np.max(img_in)
    
    if max_in == min_in: # Éviter la division par zéro si l'image est uniforme
        return img_in
        
    # Appliquer la formule de manière vectorisée
    img_out = (img_in - min_in) * ((max_out - min_out) / (max_in - min_in)) + min_out
    return img_out.astype(np.uint8)

def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Applique une correction gamma à l'image.
    
    Args:
        image (np.ndarray): Image d'entrée.
        gamma (float): Valeur gamma (< 1 éclaircit, > 1 assombrit).
        
    Returns:
        np.ndarray: Image corrigée.
    """
    # Création d'une table de correspondance (LUT) pour l'efficacité
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Application de la LUT avec la fonction optimisée d'OpenCV
    return cv2.LUT(image, table)