# # app.py (Version avec Module 5 - Morphologie)

# import streamlit as st
# import cv2
# import numpy as np
# import os
# import copy

# # Import de TOUS nos modules de traitement
# from src.processing import point_operations, spatial_filters, edge_detection, segmentation, morphology

# # --- Le reste du code jusqu'au dictionnaire 'operations' est inchangé ---
# # ... (Configuration de la page, constantes, fonctions utilitaires) ...

# st.set_page_config(page_title="Pipeline de Traitement d'Image", layout="wide")
# st.title("🔧 Pipeline de Traitement d'Image")
# st.write("Construisez une séquence d'opérations avec prévisualisation en temps réel.")

# INPUT_DIR = "data/input"
# OUTPUT_DIR = "data/output"
# os.makedirs(OUTPUT_DIR, exist_ok=True) 

# @st.cache_data
# def get_image_files():
#     files = os.listdir(INPUT_DIR)
#     return [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

# @st.cache_data
# def load_image(image_file):
#     image_path = os.path.join(INPUT_DIR, image_file)
#     return cv2.imread(image_path)

# operations = {
#     # --- Les 4 dictionnaires précédents (Opérations Ponctuelles, Filtres, Détection, Segmentation) restent les mêmes ---
#     "Opérations Ponctuelles": {
#         "Niveaux de gris": point_operations.convert_to_grayscale,
#         "Inversion (Négatif)": point_operations.invert_image,
#         "Luminosité & Contraste": point_operations.adjust_brightness_contrast,
#         "Étirement de contraste": point_operations.contrast_stretching,
#         "Correction Gamma": point_operations.apply_gamma_correction,
#         "Égalisation d'histogramme": point_operations.histogram_equalization,
#     },
#     "Filtres Spatiaux": {
#         "Filtre Moyenneur (Lissage)": spatial_filters.apply_average_filter,
#         "Filtre Gaussien (Lissage)": spatial_filters.apply_gaussian_filter,
#         "Filtre Médian (Anti-bruit)": spatial_filters.apply_median_filter,
#         "Filtre de Netteté (Sharpen)": spatial_filters.apply_sharpen_filter,
#     },
#     "Détection de Contours": {
#         "Opérateur de Sobel": edge_detection.apply_sobel,
#         "Opérateur de Prewitt": edge_detection.apply_prewitt,
#         "Opérateur de Roberts": edge_detection.apply_roberts,
#         "Opérateur Laplacien": edge_detection.apply_laplacian,
#         "Algorithme de Canny": edge_detection.apply_canny,
#     },
#     "Segmentation": {
#         "Seuillage Simple (Global)": segmentation.simple_thresholding,
#         "Seuillage d'Otsu (Automatique)": segmentation.otsu_thresholding,
#         "Seuillage Adaptatif (Local)": segmentation.adaptive_thresholding,
#     },
#     # --- NOUVEAU MODULE AJOUTÉ ---
#     "Morphologie Mathématique": {
#         "Érosion": morphology.erosion,
#         "Dilatation": morphology.dilation,
#         "Ouverture (Anti-bruit)": morphology.opening,
#         "Fermeture (Anti-trous)": morphology.closing,
#     }
# }

# # --- L'initialisation du session_state et la gestion du pipeline restent les mêmes ---
# # ...
# if 'pipeline' not in st.session_state:
#     st.session_state.pipeline = []
# if 'original_image' not in st.session_state:
#     st.session_state.original_image = None

# def apply_pipeline(image):
#     temp_image = copy.deepcopy(image)
#     for step in st.session_state.pipeline:
#         func = step['func']
#         params = step['params']
#         temp_image = func(temp_image, **params)
#     return temp_image

# def remove_step_from_pipeline(index):
#     st.session_state.pipeline.pop(index)

# st.sidebar.header("Construction du Pipeline")
# selected_image_name = st.sidebar.selectbox("1. Image de base", get_image_files())

# if st.sidebar.button("Charger Image & Vider Pipeline", use_container_width=True):
#     st.session_state.original_image = load_image(selected_image_name)
#     st.session_state.pipeline = []
#     st.rerun()

# st.sidebar.markdown("---")
# st.sidebar.subheader("Pipeline Actuel")
# if not st.session_state.pipeline:
#     st.sidebar.info("Le pipeline est vide.")
# else:
#     for i, step in enumerate(st.session_state.pipeline):
#         col1, col2 = st.sidebar.columns([3, 1])
#         col1.text(f"{i+1}. {step['name']}")
#         col2.button("🗑️", key=f"del_{i}", on_click=remove_step_from_pipeline, args=(i,), use_container_width=True)

# st.sidebar.markdown("---")
# st.sidebar.subheader("2. Ajouter une étape")
# category = st.sidebar.selectbox("Catégorie", list(operations.keys()))
# op_name = st.sidebar.selectbox("Opération", list(operations[category].keys()))

# # --- Section des paramètres MISE À JOUR pour la morphologie ---
# kwargs = {}
# with st.sidebar.expander("Paramètres de la nouvelle étape", expanded=True):
#     # Les 'if/elif' pour les opérations précédentes restent les mêmes
#     # ...
#     # --- AJOUT DE LA LOGIQUE POUR LA MORPHOLOGIE ---
#     if category == "Morphologie Mathématique":
#         kwargs['kernel_shape'] = st.selectbox("Forme du noyau", ['rectangle', 'ellipse', 'croix'], key=f"shape_{op_name}")
#         kwargs['kernel_size'] = st.slider("Taille du noyau", 3, 21, 5, step=2, key=f"ks_{op_name}")
#     # --- FIN DE L'AJOUT ---
#     elif op_name in ["Filtre Moyenneur (Lissage)", "Filtre Médian (Anti-bruit)"]:
#         kwargs['kernel_size'] = st.slider(f"Taille du noyau pour {op_name}", 3, 31, 5, step=2, key=f"ks_{op_name}")
#     # (le reste des elif est inchangé)
#     elif op_name == "Filtre Gaussien (Lissage)":
#         kwargs['kernel_size'] = st.slider(f"Taille du noyau pour {op_name}", 3, 31, 5, step=2, key=f"ks_{op_name}")
#         kwargs['sigma'] = st.slider(f"Sigma pour {op_name}", 0.1, 10.0, 1.0, key=f"sigma_{op_name}")
#     elif op_name == "Luminosité & Contraste":
#         kwargs['alpha'] = st.slider("Contraste (alpha)", 0.1, 3.0, 1.0)
#         kwargs['beta'] = st.slider("Luminosité (beta)", -100, 100, 0)
#     elif op_name == "Étirement de contraste":
#         vals = st.slider("Nouvelle plage", 0, 255, (0, 255))
#         kwargs['min_out'], kwargs['max_out'] = vals
#     elif op_name == "Correction Gamma":
#         kwargs['gamma'] = st.slider("Gamma", 0.1, 5.0, 1.0)
#     elif op_name == "Algorithme de Canny":
#         kwargs['low_threshold'] = st.slider("Seuil bas", 1, 255, 50)
#         kwargs['high_threshold'] = st.slider("Seuil haut", 1, 255, 150)
#     elif op_name == "Seuillage Simple (Global)":
#         kwargs['threshold'] = st.slider("Seuil", 0, 255, 127)
#     elif op_name == "Seuillage Adaptatif (Local)":
#         kwargs['block_size'] = st.slider("Taille du bloc", 3, 51, 11, step=2)
#         kwargs['C'] = st.slider("Constante C", -20, 20, 2)
#     else:
#         st.write("Pas de paramètres pour cette étape.")
        
# # --- Le reste du fichier (bouton 'Ajouter', et zone d'affichage) est inchangé ---
# # ...
# if st.sidebar.button("➕ Ajouter au Pipeline", use_container_width=True):
#     if st.session_state.original_image is not None:
#         op_function = operations[category][op_name]
#         st.session_state.pipeline.append({'name': op_name, 'func': op_function, 'params': kwargs})
#         st.rerun()
#     else:
#         st.sidebar.warning("Veuillez d'abord charger une image.")

# if st.session_state.original_image is not None:
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.header("1. Originale")
#         st.image(cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB), use_column_width=True)

#     image_after_pipeline = apply_pipeline(st.session_state.original_image)

#     with col2:
#         st.header("2. Prévisualisation")
#         op_function = operations[category][op_name]
#         preview_image = op_function(image_after_pipeline, **kwargs)
        
#         display_preview = preview_image
#         if len(display_preview.shape) == 3:
#             display_preview = cv2.cvtColor(display_preview, cv2.COLOR_BGR2RGB)
#         st.image(display_preview, use_column_width=True)
#         st.info(f"Prévisualisation de **{op_name}** appliqué au résultat du pipeline.")

#     with col3:
#         st.header("3. Résultat du Pipeline")
#         display_pipeline_result = image_after_pipeline
#         if len(display_pipeline_result.shape) == 3:
#             display_pipeline_result = cv2.cvtColor(display_pipeline_result, cv2.COLOR_BGR2RGB)
#         st.image(display_pipeline_result, use_column_width=True)

#         if st.button("💾 Sauvegarder ce résultat"):
#             base_name = os.path.splitext(selected_image_name)[0]
#             pipeline_name = "_".join([step['name'].split('(')[0].strip().replace(" ", "_") for step in st.session_state.pipeline]) if st.session_state.pipeline else "original"
#             output_filename = f"{base_name}_{pipeline_name.lower()}.png"
#             output_path = os.path.join(OUTPUT_DIR, output_filename)
#             cv2.imwrite(output_path, image_after_pipeline)
#             st.success(f"Image sauvegardée: `{output_path}`")
# else:
#     st.info("👋 Bienvenue ! Chargez une image depuis la barre latérale pour commencer.")    









# app.py (Version finale avec tous les modules, y compris le Domaine Fréquentiel)

import streamlit as st
import cv2
import numpy as np
import os
import copy

# Import de TOUS nos modules
from src.processing import point_operations, spatial_filters, edge_detection, segmentation, morphology, frequency

# --- Configuration et fonctions utilitaires (inchangées) ---
st.set_page_config(page_title="Boîte à Outils Pro Traitement d'Image", layout="wide")
st.title("🔧 Boîte à Outils Professionnelle de Traitement d'Image")
st.write("Construisez un pipeline d'opérations spatiales ou explorez le domaine fréquentiel.")

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@st.cache_data
def get_image_files():
    files = os.listdir(INPUT_DIR)
    return [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]

@st.cache_data
def load_image(image_file):
    image_path = os.path.join(INPUT_DIR, image_file)
    return cv2.imread(image_path)

# --- Dictionnaire complet des opérations ---
operations = {
    # ... (les 5 dictionnaires précédents restent identiques) ...
    "Opérations Ponctuelles": {"Niveaux de gris": point_operations.convert_to_grayscale, "Inversion (Négatif)": point_operations.invert_image, "Luminosité & Contraste": point_operations.adjust_brightness_contrast, "Étirement de contraste": point_operations.contrast_stretching, "Correction Gamma": point_operations.apply_gamma_correction, "Égalisation d'histogramme": point_operations.histogram_equalization},
    "Filtres Spatiaux": {"Filtre Moyenneur (Lissage)": spatial_filters.apply_average_filter, "Filtre Gaussien (Lissage)": spatial_filters.apply_gaussian_filter, "Filtre Médian (Anti-bruit)": spatial_filters.apply_median_filter, "Filtre de Netteté (Sharpen)": spatial_filters.apply_sharpen_filter},
    "Détection de Contours": {"Opérateur de Sobel": edge_detection.apply_sobel, "Opérateur de Prewitt": edge_detection.apply_prewitt, "Opérateur de Roberts": edge_detection.apply_roberts, "Opérateur Laplacien": edge_detection.apply_laplacian, "Algorithme de Canny": edge_detection.apply_canny},
    "Segmentation": {"Seuillage Simple (Global)": segmentation.simple_thresholding, "Seuillage d'Otsu (Automatique)": segmentation.otsu_thresholding, "Seuillage Adaptatif (Local)": segmentation.adaptive_thresholding},
    "Morphologie Mathématique": {"Érosion": morphology.erosion, "Dilatation": morphology.dilation, "Ouverture (Anti-bruit)": morphology.opening, "Fermeture (Anti-trous)": morphology.closing},
    # NOUVEAU MODULE
    "Domaine Fréquentiel": {
        "Filtrage Fréquentiel": frequency.frequency_domain_filter
    }
}

# --- Initialisation et gestion du pipeline (inchangé) ---
if 'pipeline' not in st.session_state: st.session_state.pipeline = []
if 'original_image' not in st.session_state: st.session_state.original_image = None
def apply_pipeline(image):
    temp_image = copy.deepcopy(image)
    for step in st.session_state.pipeline:
        temp_image = step['func'](temp_image, **step['params'])
    return temp_image
def remove_step_from_pipeline(index): st.session_state.pipeline.pop(index)

# --- Barre latérale ---
st.sidebar.header("Configuration")
selected_image_name = st.sidebar.selectbox("1. Image de base", get_image_files())

if st.sidebar.button("Charger Image & Vider Pipeline", use_container_width=True):
    st.session_state.original_image = load_image(selected_image_name)
    st.session_state.pipeline = []
    st.rerun()

st.sidebar.markdown("---")
category = st.sidebar.selectbox("2. Catégorie d'opération", list(operations.keys()))

# --- INTERFACE SPÉCIALE POUR LE DOMAINE FRÉQUENTIEL ---
if category == "Domaine Fréquentiel":
    st.sidebar.subheader("Analyse Fréquentielle")
    st.info("Le domaine fréquentiel est un mode d'analyse à part. Le pipeline n'est pas utilisé ici. Vous travaillez directement sur l'image originale.")
    
    op_name = st.sidebar.selectbox("Opération", list(operations[category].keys()))
    filter_type = st.sidebar.radio("Type de filtre", ['low-pass', 'high-pass'], captions=["Garde les basses fréquences (flou)", "Garde les hautes fréquences (détails)"])
    cutoff = st.sidebar.slider("Fréquence de coupure (rayon)", 1, 200, 30)

    if st.session_state.original_image is not None:
        filtered_image, magnitude_spectrum, mask = frequency.frequency_domain_filter(st.session_state.original_image, filter_type, cutoff)
        
        col1, col2 = st.columns(2)
        with col1:
            st.header("Image Originale")
            st.image(cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.header("Masque du Filtre")
            st.image(mask, use_column_width=True, caption=f"Rayon = {cutoff} pixels")
        with col2:
            st.header("Spectre de Fourier")
            st.image(magnitude_spectrum, use_column_width=True, caption="Visualisation logarithmique des fréquences")
            st.header("Image Filtrée")
            st.image(filtered_image, use_column_width=True, caption="Résultat après FFT inverse")
    else:
        st.warning("Veuillez charger une image pour commencer l'analyse fréquentielle.")

# --- INTERFACE STANDARD POUR LE PIPELINE (TOUTES LES AUTRES CATÉGORIES) ---
else:
    # (Le code de l'interface pipeline est quasiment inchangé)
    st.sidebar.subheader("Pipeline Actuel")
    if not st.session_state.pipeline:
        st.sidebar.info("Le pipeline est vide.")
    else:
        for i, step in enumerate(st.session_state.pipeline):
            cols = st.sidebar.columns([3, 1])
            cols[0].text(f"{i+1}. {step['name']}")
            cols[1].button("🗑️", key=f"del_{i}", on_click=remove_step_from_pipeline, args=(i,))
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ajouter une étape au pipeline")
    op_name = st.sidebar.selectbox("Opération", list(operations[category].keys()))
    
    kwargs = {}
    with st.sidebar.expander("Paramètres de la nouvelle étape", expanded=True):
        if category == "Morphologie Mathématique":
            kwargs['kernel_shape'] = st.selectbox("Forme du noyau", ['rectangle', 'ellipse', 'croix'], key=f"shape_{op_name}")
            kwargs['kernel_size'] = st.slider("Taille du noyau", 3, 21, 5, step=2, key=f"ks_{op_name}")
        # (le reste des elif est inchangé)
        elif op_name in ["Filtre Moyenneur (Lissage)", "Filtre Médian (Anti-bruit)"]: kwargs['kernel_size'] = st.slider(f"Taille du noyau pour {op_name}", 3, 31, 5, step=2, key=f"ks_{op_name}")
        elif op_name == "Filtre Gaussien (Lissage)": kwargs['kernel_size'] = st.slider(f"Taille du noyau pour {op_name}", 3, 31, 5, step=2, key=f"ks_{op_name}"); kwargs['sigma'] = st.slider(f"Sigma pour {op_name}", 0.1, 10.0, 1.0, key=f"sigma_{op_name}")
        elif op_name == "Luminosité & Contraste": kwargs['alpha'] = st.slider("Contraste (alpha)", 0.1, 3.0, 1.0); kwargs['beta'] = st.slider("Luminosité (beta)", -100, 100, 0)
        elif op_name == "Étirement de contraste": vals = st.slider("Nouvelle plage", 0, 255, (0, 255)); kwargs['min_out'], kwargs['max_out'] = vals
        elif op_name == "Correction Gamma": kwargs['gamma'] = st.slider("Gamma", 0.1, 5.0, 1.0)
        elif op_name == "Algorithme de Canny": kwargs['low_threshold'] = st.slider("Seuil bas", 1, 255, 50); kwargs['high_threshold'] = st.slider("Seuil haut", 1, 255, 150)
        elif op_name == "Seuillage Simple (Global)": kwargs['threshold'] = st.slider("Seuil", 0, 255, 127)
        elif op_name == "Seuillage Adaptatif (Local)": kwargs['block_size'] = st.slider("Taille du bloc", 3, 51, 11, step=2); kwargs['C'] = st.slider("Constante C", -20, 20, 2)
        else: st.write("Pas de paramètres pour cette étape.")
    
    if st.sidebar.button("➕ Ajouter au Pipeline", use_container_width=True):
        if st.session_state.original_image is not None:
            op_function = operations[category][op_name]
            st.session_state.pipeline.append({'name': op_name, 'func': op_function, 'params': kwargs})
            st.rerun()
        else:
            st.sidebar.warning("Veuillez d'abord charger une image.")

    if st.session_state.original_image is not None:
        col1, col2, col3 = st.columns(3)
        with col1: st.header("1. Originale"); st.image(cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB), use_column_width=True)
        image_after_pipeline = apply_pipeline(st.session_state.original_image)
        with col2:
            st.header("2. Prévisualisation")
            preview_image = operations[category][op_name](image_after_pipeline, **kwargs)
            if len(preview_image.shape) == 3: preview_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
            st.image(preview_image, use_column_width=True)
            st.info(f"Prévisualisation de **{op_name}**")
        with col3:
            st.header("3. Résultat du Pipeline")
            display_pipeline_result = image_after_pipeline
            if len(display_pipeline_result.shape) == 3: display_pipeline_result = cv2.cvtColor(display_pipeline_result, cv2.COLOR_BGR2RGB)
            st.image(display_pipeline_result, use_column_width=True)
            if st.button("💾 Sauvegarder ce résultat"):
                # (code de sauvegarde inchangé)
                base_name = os.path.splitext(selected_image_name)[0]
                pipeline_name = "_".join([step['name'].split('(')[0].strip().replace(" ", "_") for step in st.session_state.pipeline]) if st.session_state.pipeline else "original"
                output_filename = f"{base_name}_{pipeline_name.lower()}.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                cv2.imwrite(output_path, image_after_pipeline)
                st.success(f"Image sauvegardée: `{output_path}`")
    else:
        st.info("👋 Bienvenue ! Chargez une image depuis la barre latérale pour commencer.")
