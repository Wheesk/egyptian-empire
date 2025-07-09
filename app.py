import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import openai

# Load FAISS index and chunks
@st.cache_resource
def load_faiss_and_chunks():
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index("faiss_index.index")
    return chunks, index

chunks, index = load_faiss_and_chunks()

# Embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Set up OpenAI client using Streamlit secrets
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_top_chunks(query, k=3):
    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def ask_openai(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful historian assistant with expert knowledge about Ancient Egypt.
Use the context below to answer the question in 3–4 clear sentences.

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

st.title("Ask Ancient Egypt (RAG Q&A Demo)")

user_question = st.text_input("Ask a question about Ancient Egypt:")

if user_question:
    with st.spinner("Searching and generating answer..."):
        context = get_top_chunks(user_question, k=3)
        st.markdown("**Retrieved context:**")
        for i, chunk in enumerate(context):
            st.markdown(f"<details><summary>Chunk {i+1}</summary><pre>{chunk[:1000]}</pre></details>", unsafe_allow_html=True)
        answer = ask_openai(user_question, context)
        st.markdown("**📘 Answer:**")
        st.success(answer)
