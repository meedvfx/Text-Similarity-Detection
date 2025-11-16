import streamlit as st
import re  # Importation de la bibliothèque RegEx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_text(text):
    """
    Applique le prétraitement au texte avec RegEx (sans NLTK) :
    1. Met en minuscules
    2. Supprime la ponctuation et les chiffres
    """
    # 1. Minuscules
    text_lower = text.lower()

    # 2. Suppression de tout ce qui n'est pas une lettre ou un espace
    # C'est ici que 're' est utilisé
    text_cleaned = re.sub(r'[^a-z\s]', '', text_lower)

    # 3. Suppression des espaces multiples
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()

    return text_cleaned


st.set_page_config(page_title="Détecteur de Similarité", layout="wide")
st.title("🔎 Détecteur de Similarité de Texte (Plagiat)")
st.write("Comparaison de modèles pour la détection de similarité.")

st.divider()

# --- 1. CHOIX DU MODÈLE ---
st.header("1. Choisissez votre modèle")
model_choice = st.radio(
    "Sélectionnez la méthode d'analyse :",
    ('TF-IDF', 'Sentence-BERT (S-BERT)', 'LSTM'),
    horizontal=True,
    key="model_select",
    help="Choisissez l'algorithme à utiliser pour la comparaison."
)

# --- 2. OPTIONS CONDITIONNELLES (POUR TF-IDF) ---
# Cette section n'apparaîtra que si 'TF-IDF' est sélectionné
ngram_tuple = (1, 1)  # Valeur par défaut
if model_choice == 'TF-IDF':
    st.subheader("Options TF-IDF")
    ngram_max = st.selectbox(
        "Taille maximale des N-grams :",
        (1, 2, 3, 4),
        index=0,
        format_func=lambda x: f"{x} (jusqu'à {x}-grams)" if x > 1 else f"{x} (mots seuls)",
    )
    # TfidfVectorizer attend un tuple (min_n, max_n)
    ngram_tuple = (1, ngram_max)

st.divider()

# --- 3. ENTRÉE DES TEXTES ---
st.header("2. Entrez vos textes")
col1, col2 = st.columns(2)

with col1:
    st.header("Texte 1")
    text1 = st.text_area("Collez votre premier texte ici :", height=300, key="txt1")

with col2:
    st.header("Texte 2")
    text2 = st.text_area("Collez votre deuxième texte ici :", height=300, key="txt2")

st.divider()

# --- 4. BOUTON ET LOGIQUE DE CALCUL ---
if st.button("Calculer la Similarité", type="primary"):

    # Vérifier si les textes sont vides
    if not (text1.strip() and text2.strip()):
        st.warning("Veuillez entrer du texte dans les deux boîtes.")

    else:
        # --- LOGIQUE DE ROUTAGE (selon le modèle choisi) ---

        if model_choice == 'TF-IDF':
            st.subheader(f"Résultats (Modèle : TF-IDF avec N-grams={ngram_tuple})")
            try:
                # 1. Prétraitement
                st.write("Prétraitement en cours...")
                proc_text1 = preprocess_text(text1)
                proc_text2 = preprocess_text(text2)
                documents = [proc_text1, proc_text2]

                # 2. Vectorisation (AVEC N-GRAMS)
                st.write(f"Vectorisation (TF-IDF) avec n-grams de {ngram_tuple}...")

                # C'est ici que l'option n-gram est passée au modèle
                vectorizer = TfidfVectorizer(ngram_range=ngram_tuple)
                tfidf_matrix = vectorizer.fit_transform(documents)

                # 3. Modélisation (Calcul de la similarité cosinus)
                st.write("Calcul de la similarité cosinus...")
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                similarity_score = cosine_sim[0][0]

                # Affichage des résultats
                st.divider()
                score_percent = similarity_score * 100
                st.metric(
                    label="Score de Similarité (TF-IDF)",
                    value=f"{score_percent:.2f} %"
                )
                st.progress(similarity_score)

                if similarity_score > 0.8:
                    st.error("🚨 **Alerte :** Similarité très élevée. Risque de plagiat.")
                elif similarity_score > 0.5:
                    st.warning("⚠️ **Avertissement :** Similarité notable.")
                else:
                    st.success("✅ **OK :** Les textes semblent différents.")

            except ValueError:
                st.warning("Les textes sont vides après nettoyage. Impossible de calculer la similarité.")

        # --- Blocs pour les futurs modèles ---

        elif model_choice == 'Sentence-BERT (S-BERT)':
            st.subheader("Résultats (Modèle : Sentence-BERT)")
            st.info("🚧 Ce modèle n'est pas encore développé.")
            st.write(
                "L'implémentation de Sentence-BERT (S-BERT) viendra ici. Ce modèle est excellent pour comprendre le *sens* sémantique des phrases.")

        elif model_choice == 'LSTM':
            st.subheader("Résultats (Modèle : LSTM)")
            st.info("🚧 Ce modèle n'est pas encore développé.")
            st.write("L'implémentation du modèle LSTM siamois viendra ici.")
