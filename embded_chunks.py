import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

#  Load saved chunks
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

#  Load local embedding model 
model = SentenceTransformer("all-MiniLM-L6-v2")  

#  Generate embeddings for all chunks
print(" Generating embeddings locally...")
embeddings = model.encode(chunks, show_progress_bar=True)

#  Build FAISS index
embedding_matrix = np.array(embeddings).astype("float32")
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

#  Save index for querying later
faiss.write_index(index, "faiss_index.index")
print(" Local embeddings complete and FAISS index saved.")
