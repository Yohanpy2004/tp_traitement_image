# src/processing/frequency.py

import cv2
import numpy as np
from .point_operations import convert_to_grayscale

def frequency_domain_filter(image: np.ndarray, filter_type: str = 'low-pass', cutoff: int = 30) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Applique un filtrage dans le domaine fréquentiel (Passe-bas ou Passe-haut).
    
    Args:
        image (np.ndarray): Image d'entrée.
        filter_type (str): 'low-pass' ou 'high-pass'.
        cutoff (int): Fréquence de coupure (rayon du masque circulaire).
        
    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray): 
            - L'image filtrée.
            - La visualisation du spectre de magnitude (logarithmique).
            - La visualisation du masque du filtre.
    """
    gray_image = convert_to_grayscale(image)

    # 1. Padding pour des dimensions optimales pour la FFT
    rows, cols = gray_image.shape
    m, n = cv2.getOptimalDFTSize(rows), cv2.getOptimalDFTSize(cols)
    padded = cv2.copyMakeBorder(gray_image, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 2. Appliquer la Transformée de Fourier
    # On passe d'une image réelle à un plan complexe
    dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # 3. Déplacer le quadrant de fréquence zéro au centre du spectre
    dft_shift = np.fft.fftshift(dft)

    # 4. Créer le masque de filtrage (cercle au centre)
    center_row, center_col = m // 2, n // 2
    mask = np.zeros((m, n, 2), np.uint8) # Masque a 2 canaux pour réel et imaginaire
    
    # Création d'une grille de coordonnées
    y, x = np.ogrid[:m, :n]
    mask_area = (x - center_col)**2 + (y - center_row)**2 <= cutoff**2
    
    if filter_type == 'low-pass':
        mask[mask_area] = 1
    elif filter_type == 'high-pass':
        mask[~mask_area] = 1
    else:
        raise ValueError("Type de filtre inconnu")

    # 5. Visualisation du spectre de magnitude (pour l'affichage)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    magnitude_spectrum_visual = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 6. Appliquer le masque en multipliant dans le domaine fréquentiel
    fshift = dft_shift * mask

    # 7. Inverser le décalage du quadrant
    f_ishift = np.fft.ifftshift(fshift)
    
    # 8. Appliquer la Transformée de Fourier Inverse
    img_back = cv2.idft(f_ishift)
    
    # 9. Récupérer la partie réelle et recadrer l'image
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    filtered_image = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    filtered_image = filtered_image[:rows, :cols] # Recadrage à la taille originale

    # Visualisation du masque (convertir en image affichable)
    mask_visual = mask[:, :, 0] * 255 

    return filtered_image, magnitude_spectrum_visual, mask_visual