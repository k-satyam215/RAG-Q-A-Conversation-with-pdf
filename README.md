# Conversational RAG with PDFs & Chat History 📘🤖

This project is a **Conversational Retrieval-Augmented Generation (RAG) system** built using **Streamlit, LangChain, Chroma, and Groq LLMs**.  
It allows users to **upload one or more PDF files** and **chat with their content**, while maintaining **multi-turn conversation history**.

The system intelligently reformulates follow-up questions using chat history and retrieves relevant PDF context to generate concise, grounded answers.

---

## 🎯 Objective

The goal of this project is to:
- Build a **chat-based Q&A system over PDFs**
- Demonstrate **history-aware conversational RAG**
- Combine embeddings, vector search, and LLM reasoning
- Maintain session-based chat history for contextual understanding

---

## 🚀 Key Features

### 📂 PDF Upload & Processing
- Supports **multiple PDF uploads**
- Extracts text using `PyPDFLoader`
- Splits documents into overlapping chunks for better retrieval

---

### 🧠 Semantic Search with Embeddings
- Uses **HuggingFace sentence-transformers**
- Model: `all-MiniLM-L6-v2`
- Creates a **Chroma vector store** for fast similarity search

---

### 💬 Conversational RAG (History-Aware)
- Reformulates follow-up questions using chat history
- Uses `create_history_aware_retriever`
- Enables natural multi-turn conversations over documents

---

### 🤖 LLM-Powered Answers
- Powered by **Groq-hosted LLaMA 3.1 (8B Instant)**
- Uses retrieved PDF context only
- Produces **concise answers (max 3 sentences)**

---

### 🗂️ Session-Based Chat Memory
- Multiple conversation sessions supported
- Chat history stored in Streamlit session state
- Each session maintains independent context

---

### 🖥️ Interactive Streamlit UI
- PDF uploader
- Secure Groq API key input
- Session ID control
- Real-time Q&A interface

---

## 🧠 How It Works

1. User uploads one or more PDF files
2. PDFs are loaded and split into chunks
3. Chunks are embedded and stored in Chroma
4. User asks a question
5. Chat history is used to reformulate the question (if needed)
6. Relevant chunks are retrieved
7. LLM generates a concise answer grounded in PDF content
8. Chat history is updated and preserved per session

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Groq API**
- **LLaMA 3.1**
- **HuggingFace Embeddings**
- **Chroma Vector Store**
- **PyPDFLoader**



