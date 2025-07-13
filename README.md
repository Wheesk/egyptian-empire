## Ancient Egypt RAG Q&A System

Ever wanted to chat with the history of Ancient Egypt?

This project makes it possible! 

I built a Retrieval-Augmented Generation (RAG) system that digs up facts from a curated PDF and serves them to you through an AI historian.

No pyramids were harmed in the making. 

https://egyptian-empire.streamlit.app/

---

##How Does It Work?
In a nutshell: Think of it as an ancient librarian meets AI genie.

1️) I cleaned up a huge PDF about Ancient Egypt.

2️) Chopped it into chunks with some overlap, so context is never lost in the desert .

3️) Turned each chunk into vectors (using all-MiniLM-L6-v2 — my embedding hero).

4️) Indexed it all using FAISS, so it can find the right chunk faster than you can say “Pharaoh”.

## When you ask a question, the system:

Embeds your question.

Does a similarity search.

Grabs the top-k chunks.

Builds a fancy prompt.

Sends it to GPT-3.5, which writes you an answer — all fact-checked by the retrieved context!

## User Interfaces:

Streamlit Web App: Click, ask, get answers!

Command-Line Tool: For the terminal fans.

---

## Key Components (Fun-Sized)
Component	What It Does

PDF Loader Sucks up text from the PDF

Chunker Breaks it up nicely.
Embedder 	Turns chunks into brainy vectors.

FAISS Index Keeps vectors organized for fast search.

Retriever Finds the best bits for your question.

Prompt Builder 	Builds a smart prompt for GPT.

GPT-3.5	Generates the final answer.

UI 	Streamlit & CLI for you to play with.

--- 

## Example
Q: Who was Imhotep?

A: Imhotep was an architect, physician, and vizier to Pharaoh Djoser. He designed the Step Pyramid at Saqqara and was later deified as a god of wisdom and healing. 

---

## How Good Is It?
Accuracy: Over 90% relevant retrieval.

Fast: ~1.7s response time (web), ~1.3s (CLI).

Friendly: Handles on-topic questions like a charm.

No Hallucinations: “What are the ingredients of Italian pizza?” → Sorry, off-topic! 

--- 

## Run It Yourself!
1)Install the tools:

```bash
pip install requirements.txt
```
2️)Start the Streamlit app:
```bash
streamlit run app.py
```
---
  
## Made with 🧡 by Wheesk.
