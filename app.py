import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import openai

# -- Custom CSS for cleaner look --
st.markdown("""
    <style>
        .main {background-color: #fcfcfc;}
        .stButton>button {background-color:#e5d8b6;}
        .stTextInput>div>div>input {background-color:#fff3e6;}
        .chunk-box {background: #f7f2e5; border-radius: 8px; padding: 8px 16px;}
        .answer-box {background: #e9fbe5; border-radius: 10px; padding: 12px 20px; font-size: 1.08em;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: #7c552f; margin-bottom:0;'>Ancient Egypt RAG Q&A</h1>", unsafe_allow_html=True)
st.markdown("<span style='color: #98815b;'>Ask questions about Ancient Egypt and get AI-powered answers, grounded in our PDF knowledge base!</span>", unsafe_allow_html=True)
st.markdown("---")

# Example question
EXAMPLE_Q = "Who was Cleopatra VII and why is she famous?"

if st.button("Try an example question!"):
    st.session_state.user_question = EXAMPLE_Q

user_question = st.text_input("Ask a question about Ancient Egypt:", value=st.session_state.get("user_question", ""))

# -- Load FAISS and chunks --
@st.cache_resource
def load_faiss_and_chunks():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("faiss_index.index")
    return chunks, index

chunks, index = load_faiss_and_chunks()

# -- Load embedding model --
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = load_embedding_model()

# -- Set up OpenAI client using Streamlit secrets --
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_top_chunks(query, k=3):
    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def ask_openai(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful historian assistant with expert knowledge about Ancient Egypt.
Use ONLY the context below to answer the question in 3–4 clear sentences.

Context:
{context}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful historian assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=256,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# -- UI Interaction --
if user_question:
    with st.spinner("🔎 Searching and generating your answer..."):
        context = get_top_chunks(user_question, k=3)
        st.markdown("#### 🔗 **Top retrieved context from the PDF:**")
        for i, chunk in enumerate(context):
            with st.expander(f"Chunk {i+1}"):
                st.markdown(f"<div class='chunk-box'>{chunk[:1500]}</div>", unsafe_allow_html=True)
        answer = ask_openai(user_question, context)
        st.markdown("#### 📘 **AI Answer:**")
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<center><span style='font-size: 0.95em; color: #888;'>Built by [Your Name], powered by Streamlit, FAISS, Sentence Transformers, and OpenAI GPT-3.5.</span></center>", unsafe_allow_html=True)
