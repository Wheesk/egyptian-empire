import faiss
import pickle
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
import os
import openai


load_dotenv()  
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  

# Load FAISS index and chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
index = faiss.read_index("faiss_index.index")

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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

if __name__ == "__main__":
    print("RAG Q&A with OpenAI GPT-3.5")
    while True:
        query = input("\nAsk a question about Ancient Egypt (or type 'exit'): ")
        if query.lower() == "exit":
            break
        context = get_top_chunks(query, k=3)
        print("\n[DEBUG] Context chunks used for answer:")
        for i, chunk in enumerate(context):
            print(f"\n--- Chunk {i+1} ---\n{chunk[:500]}...\n")
        answer = ask_openai(query, context)
        print("\n📘 Answer:")
        print(answer)
