import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import os # <-- AJOUTER CET IMPORT

# --- DÉBUT DE LA NOUVELLE SOLUTION ---

# 1. Définir un chemin local pour les données NLTK
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")

# 2. Créer le dossier s'il n'existe pas
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)

# 3. Dire à NLTK de TOUJOURS chercher les données ici
nltk.data.path.append(NLTK_DATA_DIR)

@st.cache_resource  # Met en cache cette fonction
def download_nltk_resources(download_dir):
    """Télécharge les paquets NLTK requis dans un dossier spécifique."""
    try:
        # Spécifier le dossier de téléchargement !
        nltk.download('punkt', download_dir=download_dir)
        nltk.download('stopwords', download_dir=download_dir)
        nltk.download('wordnet', download_dir=download_dir)
        print(f"Téléchargement NLTK réussi dans {download_dir}")
        return True
    except Exception as e:
        print(f"Erreur lors du téléchargement NLTK : {e}")
        return False

# Exécute la fonction de téléchargement au démarrage
NLTK_READY = download_nltk_resources(NLTK_DATA_DIR)

# --- FIN DE LA NOUVELLE SOLUTION ---

def preprocess_text(text):
    """
    Applique le prétraitement au texte :
    1. Minuscules
    2. Tokenisation
    3. Suppression de la ponctuation
    4. Suppression des stopwords
    5. Lemmatisation
    """
    # 1. Minuscules
    text_lower = text.lower()

    # 2. Tokenisation
    tokens = word_tokenize(text_lower)

    # 3. Suppression de la ponctuation
    tokens = [w for w in tokens if w not in string.punctuation]

    # 4. Suppression des stopwords (anglais par défaut, changez pour 'french')
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [w for w in tokens if w not in stop_words and w.isalnum()]

    # 5. Lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(t) for t in cleaned_tokens]

    return " ".join(lemmatized_tokens)


st.set_page_config(page_title="Détecteur de Similarité", layout="wide")
st.title("🔎 Détecteur de Similarité de Texte (Plagiat)")
st.write("Basé sur TF-IDF et la Similarité Cosinus")

# --- AJOUTER CETTE VÉRIFICATION ---
if NLTK_READY:
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
            # 1. Prétraitement du texte
            st.write("Prétraitement en cours...")
            proc_text1 = preprocess_text(text1)
            proc_text2 = preprocess_text(text2)

            documents = [proc_text1, proc_text2]

            # 2. Vectorisation (TF-IDF)
            st.write("Vectorisation (TF-IDF)...")
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents)

            # 3. Modélisation (Calcul de la similarité cosinus)
            st.write("Calcul de la similarité cosinus...")
            # On compare le vecteur 0 (texte 1) au vecteur 1 (texte 2)
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

        else:
            st.warning("Veuillez entrer du texte dans les deux boîtes.")

else:
    # Si NLTK n'a pas pu se télécharger, afficher une erreur
    st.error("Erreur critique : L'application n'a pas pu télécharger les ressources NLTK nécessaires pour fonctionner.")
    st.stop()