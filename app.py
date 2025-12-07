import streamlit as st
import re
import pypdf  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


if 'text1_content' not in st.session_state:
    st.session_state.text1_content = ""
if 'text2_content' not in st.session_state:
    st.session_state.text2_content = ""


def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Erreur lors de la lecture du PDF : {e}")
        return ""



def update_text1_from_pdf():
    uploaded_file = st.session_state.pdf1_uploader
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        st.session_state.text1_content = text


def update_text2_from_pdf():
    uploaded_file = st.session_state.pdf2_uploader
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        st.session_state.text2_content = text


def preprocess_text(text):
    text_lower = text.lower()
    text_cleaned = re.sub(r'[^a-z\s]', '', text_lower)
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
    return text_cleaned


@st.cache_resource
def load_sbert_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


sbert_model = load_sbert_model()

st.set_page_config(page_title="Détecteur de Similarité", layout="wide")
st.title("🔎 Détecteur de Similarité de Texte (Plagiat)")
st.write("Comparez deux textes par copier-coller ou en important des fichiers PDF.")

st.divider()

st.header("1. Choisissez votre modèle")
model_choice = st.radio(
    "Sélectionnez la méthode d'analyse :",
    ('TF-IDF', 'Sentence-BERT (S-BERT)', 'LSTM'),
    horizontal=True
)

ngram_tuple = (1, 1)
if model_choice == 'TF-IDF':
    st.subheader("Options TF-IDF")
    ngram_max = st.selectbox(
        "Taille maximale des N-grams :",
        (1, 2, 3, 4),
        format_func=lambda x: f"{x} (jusqu'à {x}-grams)" if x > 1 else f"{x} (mots seuls)"
    )
    ngram_tuple = (1, ngram_max)

st.divider()

st.header("2. Importez ou collez vos textes")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Document 1")
    st.file_uploader(
        "Importer un PDF (optionnel)",
        type="pdf",
        key="pdf1_uploader",
        on_change=update_text1_from_pdf
    )
    text1 = st.text_area(
        "Contenu du texte 1 :",
        height=300,
        key="text1_content"  
    )

with col2:
    st.subheader("Document 2")
    st.file_uploader(
        "Importer un PDF (optionnel)",
        type="pdf",
        key="pdf2_uploader",
        on_change=update_text2_from_pdf
    )
    text2 = st.text_area(
        "Contenu du texte 2 :",
        height=300,
        key="text2_content"  
    )

st.divider()

# --- 3. CALCUL ---
if st.button("Calculer la Similarité", type="primary"):


    content1 = text1.strip()
    content2 = text2.strip()

    if not (content1 and content2):
        st.warning("Veuillez fournir du texte pour les deux documents.")

    else:
        if model_choice == 'TF-IDF':
            st.subheader(f"Résultats (TF-IDF : {ngram_tuple})")
            try:
                proc_text1 = preprocess_text(content1)
                proc_text2 = preprocess_text(content2)
                documents = [proc_text1, proc_text2]

                vectorizer = TfidfVectorizer(ngram_range=ngram_tuple)
                tfidf_matrix = vectorizer.fit_transform(documents)

                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                similarity_score = cosine_sim[0][0]

                st.divider()
                st.metric("Score de Similarité", f"{similarity_score * 100:.2f} %")
                st.progress(similarity_score)

                if similarity_score > 0.8:
                    st.error("🚨 Risque élevé de plagiat.")
                elif similarity_score > 0.5:
                    st.warning("⚠️ Similarité notable.")
                else:
                    st.success("✅ Textes différents.")

            except ValueError:
                st.warning("Erreur : Textes vides après nettoyage.")

        elif model_choice == 'Sentence-BERT (S-BERT)':
            st.subheader("Résultats (Sentence-BERT)")

            documents = [content1, content2]
            embeddings = sbert_model.encode(documents)
            cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
            similarity_score = cosine_sim[0][0].item()

            st.divider()
            st.metric("Score Sémantique", f"{similarity_score * 100:.2f} %")
            st.progress(similarity_score)

            if similarity_score > 0.8:
                st.error("🚨 Sens très proche.")
            elif similarity_score > 0.5:
                st.warning("⚠️ Sens similaire.")
            else:
                st.success("✅ Sens différent.")

        elif model_choice == 'LSTM':
            st.info("🚧 Modèle LSTM en cours de développement.")
