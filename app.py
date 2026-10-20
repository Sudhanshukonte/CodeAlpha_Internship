import json
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "data" / "faqs.json"
SIMILARITY_THRESHOLD = 0.20

@st.cache_data
def load_faqs():
    with FAQ_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list) or not data:
        raise ValueError("The FAQ file must contain a non-empty JSON list.")
    for item in data:
        if "question" not in item or "answer" not in item:
            raise ValueError("Every FAQ entry needs a question and an answer.")
    return data

@st.cache_resource
def build_search_index(questions):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
    )
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix

def find_answer(user_question, faqs, vectorizer, faq_matrix):
    query_vector = vectorizer.transform([user_question])
    scores = cosine_similarity(query_vector, faq_matrix).flatten()
    best_index = int(scores.argmax())
    best_score = float(scores[best_index])
    if best_score < SIMILARITY_THRESHOLD:
        return (
            "I couldn't find a confident match in my FAQ knowledge base. "
            "Try rephrasing your question or ask about a topic covered in the FAQs.",
            best_score,
            None,
        )
    return faqs[best_index]["answer"], best_score, faqs[best_index]["question"]

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")
st.title("💬 FAQ Chatbot")
st.caption("A lightweight NLP chatbot powered by TF-IDF and cosine similarity.")
st.write("Ask a question about Python, programming, or this demo.")

try:
    faqs = load_faqs()
    vectorizer, faq_matrix = build_search_index([item["question"] for item in faqs])
except (OSError, json.JSONDecodeError, ValueError) as error:
    st.error(f"Could not load the FAQ dataset: {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me a question from the FAQ knowledge base."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Type your question…")
if prompt:
    prompt = prompt.strip()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        answer, score, matched_question = find_answer(prompt, faqs, vectorizer, faq_matrix)
        with st.chat_message("assistant"):
            st.markdown(answer)
            if matched_question:
                st.caption(f"Matched FAQ: “{matched_question}” · similarity: {score:.2f}")
        st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.header("About this demo")
    st.write("The chatbot compares your question with stored FAQ questions.")
    st.write("It uses TF-IDF to represent text and cosine similarity to rank matches.")
    st.write("This is retrieval-based: it selects an answer from the dataset rather than generating new facts.")
    if st.button("Clear conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Conversation cleared. What would you like to know?"}
        ]
        st.rerun()
    with st.expander("Sample questions"):
        for item in faqs[:8]:
            st.write(f"• {item['question']}")
