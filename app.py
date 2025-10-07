# -------------------------------
# Infosys PaperIQ: Classical NLP Version (Chat + Download History, Single Detailed Answer)
# -------------------------------

import streamlit as st
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import PyPDF2
import docx2txt
from textblob import TextBlob

# -------------------------------
# NLTK downloads
# -------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------------
# Helper Functions
# -------------------------------

def extract_text(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        pdf = PyPDF2.PdfReader(file_path)
        for page in pdf.pages:
            text += page.extract_text()
    elif file_path.endswith(".docx"):
        text = docx2txt.process(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9. ]', '', text)
    sentences = sent_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    processed_sentences = []
    for sent in sentences:
        words = word_tokenize(sent)
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        processed_sentences.append(" ".join(words))
    return sentences, processed_sentences

def summarize(sentences, processed_sentences, n=5):
    words = " ".join(processed_sentences).split()
    freq = Counter(words)
    sentence_scores = {}
    for i, sent in enumerate(processed_sentences):
        score = sum(freq.get(word,0) for word in sent.split())
        sentence_scores[i] = score
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:n]
    summary = [sentences[i] for i in top_sentences]
    return " ".join(summary)

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords[:top_n]

# -------------------------------
# Single detailed answer for chat
# -------------------------------
def answer_question_single_detailed(text, question):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer(stop_words='english')
    all_sentences = sentences + [question]
    tfidf_matrix = vectorizer.fit_transform(all_sentences)
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    top_idx = similarity.argsort()[0][-1]  # only the most relevant sentence
    # Expand the answer by including neighboring sentences for context
    start = max(0, top_idx-1)
    end = min(len(sentences), top_idx+2)
    detailed_answer = " ".join(sentences[start:end])
    return detailed_answer

def lda_topics(sentences, n_topics=3, n_words=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    topics = []
    for i, topic in enumerate(lda.components_):
        topic_words = [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-n_words:]]
        topics.append(f"Topic {i+1}: " + ", ".join(topic_words))
    return topics

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Research Navigator", layout="wide")
st.title("📄 Research Navigator - Chat & Research Analyzer")

# -------------------------------
# Session State
# -------------------------------
for key in ["uploaded_file", "extracted_text", "sentences", "processed_sentences", "chat_history", "download_history"]:
    if key not in st.session_state:
        st.session_state[key] = [] if "history" in key else None

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Select Option", ["Upload & Preview", "Summary & Keywords", "Topics & Sentiment", "Q&A & Chat", "Download History"])

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.sidebar.file_uploader("Upload PDF/DOCX/TXT", type=["pdf", "docx", "txt"])
if uploaded_file:
    save_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.uploaded_file = uploaded_file
    st.session_state.extracted_text = extract_text(save_path)
    st.session_state.sentences, st.session_state.processed_sentences = preprocess_text(st.session_state.extracted_text)

# -------------------------------
# Panels
# -------------------------------
if menu == "Upload & Preview":
    if st.session_state.extracted_text:
        st.subheader("✅ Extracted Text (Preview)")
        st.text(st.session_state.extracted_text[:1000] + "...")
    else:
        st.info("Please upload a document.")

elif menu == "Summary & Keywords":
    if st.session_state.extracted_text:
        summary = summarize(st.session_state.sentences, st.session_state.processed_sentences, n=5)
        keywords = extract_keywords(st.session_state.extracted_text)
        st.subheader("📝 Extractive Summary")
        st.write(summary)
        st.subheader("🔑 Keywords")
        st.write(", ".join(keywords))
        if ("Summary", summary) not in st.session_state.download_history:
            st.session_state.download_history.append(("Summary", summary))
        if st.button("Download Summary as TXT"):
            file_name = f"{st.session_state.uploaded_file.name}_summary.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(summary)
            st.success(f"Summary saved as {file_name}")
    else:
        st.info("Upload a document first.")

elif menu == "Topics & Sentiment":
    if st.session_state.extracted_text:
        lda_res = lda_topics(st.session_state.sentences)
        st.subheader("📚 LDA Topics")
        for topic in lda_res:
            st.write(topic)
        polarity, subjectivity = analyze_sentiment(st.session_state.extracted_text)
        st.subheader("💡 Sentiment Analysis")
        st.write(f"Polarity: {polarity:.2f} (-1 Negative, 1 Positive)")
        st.write(f"Subjectivity: {subjectivity:.2f} (0 Objective, 1 Subjective)")
    else:
        st.info("Upload a document first.")

elif menu == "Q&A & Chat":
    if st.session_state.extracted_text:
        st.subheader("💬 Conversational Q&A")
        user_input = st.text_input("Ask a question:")
        if user_input:
            answer_text = answer_question_single_detailed(st.session_state.extracted_text, user_input)
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", answer_text))
        for sender, message in st.session_state.chat_history:
            st.markdown(f"**{sender}:** {message}")
    else:
        st.info("Upload a document first.")

elif menu == "Download History":
    st.subheader("📥 Download History")
    if st.session_state.download_history:
        for item_type, item_content in st.session_state.download_history:
            if item_type == "Summary":
                st.markdown(f"**Summary:** {item_content[:200]}...")
    else:
        st.info("No downloads yet.")
