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
st.write("Basé sur TF-IDF et la Similarité Cosinus")

# Créer deux colonnes pour les boîtes de texte
col1, col2 = st.columns(2)

with col1:
    st.header("Texte 1")
    text1 = st.text_area("Collez votre premier texte ici :", height=300, key="txt1")

with col2:
    st.header("Texte 2")
    text2 = st.text_area("Collez votre deuxième texte ici :", height=300, key="txt2")

# Bouton pour lancer le calcul
if st.button("Calculer la Similarité", type="primary"):
    if text1.strip() and text2.strip():
        # 1. Prétraitement du texte (maintenant avec 're')
        st.write("Prétraitement en cours...")
        proc_text1 = preprocess_text(text1)
        proc_text2 = preprocess_text(text2)

        documents = [proc_text1, proc_text2]

        # 2. Vectorisation (TF-IDF)
        st.write("Vectorisation (TF-IDF)...")
        # TfidfVectorizer va maintenant travailler sur le texte déjà nettoyé
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        try:
            # 3. Modélisation (Calcul de la similarité cosinus)
            st.write("Calcul de la similarité cosinus...")
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

            # Le résultat est une matrice, on prend le premier (et seul) élément
            similarity_score = cosine_sim[0][0]

            # Affichage des résultats
            st.divider()
            st.subheader("Résultats")

            # Formater le score en pourcentage
            score_percent = similarity_score * 100

            st.metric(
                label="Score de Similarité",
                value=f"{score_percent:.2f} %"
            )

            st.progress(similarity_score)

            if similarity_score > 0.8:
                st.error("🚨 **Alerte :** Similarité très élevée. Risque de plagiat.")
            elif similarity_score > 0.5:
                st.warning("⚠️ **Avertissement :** Similarité notable. Les textes partagent un vocabulaire commun.")
            else:
                st.success("✅ **OK :** Les textes semblent différents.")

        except ValueError:
            # Gère le cas où les textes sont vides après prétraitement
            # (par exemple, si l'utilisateur ne met que des chiffres ou de la ponctuation)
            st.warning("Les textes sont vides après nettoyage. Impossible de calculer la similarité.")

    else:
        st.warning("Veuillez entrer du texte dans les deux boîtes.")