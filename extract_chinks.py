import fitz
import pandas as pd
import re

doc = fitz.open("Ancient Egypt.pdf")

all_text = ""
for page in doc:
    all_text += page.get_text()




raw_paragraphs = re.split(r'\n\s*\n|(?<=\.)\s{2,}', all_text)
paragraphs = [p.strip() for p in raw_paragraphs if len(p.strip()) > 100]


def create_chunks(paragraphs, max_words = 400, overlap = 50):
    chunks = []
    buffer = []
    buffer_len = 0

    for para in paragraphs:
        para_words = para.split()
        if buffer_len + len(para_words) <= max_words:
            buffer.extend(para_words)
            buffer_len += len(para_words)
        else:
            chunks.append(" ".join(buffer))
            buffer = buffer[-overlap:] + para_words
            buffer_len = len(buffer)

    if buffer:
        chunks.append(" ".join(buffer))
    return chunks

chunks = create_chunks(paragraphs)

for i, chunk in enumerate(chunks[:5]):
    print(f"\n-- Chunk {i+1} ---\n{chunk[:500]}...\n")

print(f"\n✅ Total chunks created: {len(chunks)}")

import pickle

# Only run this after you’ve confirmed chunks are created (e.g. 31 chunks)
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ chunks.pkl saved successfully.")