import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import openai

# -- Custom CSS for cleaner look --

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1705628080778-f86b2f90a114?q=80&w=2338&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    .stApp {background: rgba(0,0,0,0.5);}
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown("""
    <style>
        .main {background-color: #181818;}
        .stButton>button {background-color:#c0b283; color: #222;}
        .stTextInput>div>div>input {background-color:#23272e; color: #f0e9d2;}
        .chunk-box {background: #23392d; color: #c5f6c5; border-radius: 8px; padding: 8px 16px;}
        .answer-box {background: #23392d; color: #c5f6c5; border-radius: 10px; padding: 12px 20px; font-size: 1.08em;}
        /* Make markdown answer text visible and not faded */
        .answer-box p, .answer-box strong, .answer-box span { color: #c5f6c5 !important; }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='color: #D5BABA; margin-bottom:0;'>Ancient Egypt Empire RAG Q&A</h1>", unsafe_allow_html=True)
st.markdown("<span style='color: #D5BABA;'>Ask questions about Ancient Egypt and get AI-powered answers, grounded in our PDF knowledge base!</span>", unsafe_allow_html=True)
st.markdown("---")

# Example question
EXAMPLE_QUESTIONS = [
    "Who was Cleopatra VII and why is she famous?",
    "How did the Nile River influence Ancient Egypt?",
    "What are pyramids and why were they built?",
    "Who was Imhotep?",
    "Describe the process of mummification.",
    "What was the role of women in Ancient Egyptian society?",
    "Who was the first king in the Egyptian empire?"
]

st.markdown("#### Need inspiration? Pick an example question:")

# Dropdown selectbox for example questions
selected_example = st.selectbox(
    "Choose an example question:",
    ["Select a question..."] + EXAMPLE_QUESTIONS
)

if selected_example and selected_example != "Select a question...":
    st.session_state["user_question"] = selected_example

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
st.markdown("<center><span style='font-size: 0.95em; color: #888;'>Built by Wheesk, powered by Streamlit, FAISS, Sentence Transformers, and OpenAI GPT-3.5.</span></center>", unsafe_allow_html=True)
