## RAG Q&A Conversation With PDF Including Chat History

import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# -----------------------------
# 1. Load Environment Variables
# -----------------------------
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# -----------------------------
# 2. Initialize HuggingFace Embeddings
# -----------------------------
# Add model_kwargs to avoid meta tensor error on PyTorch 2.5+
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # Safe option
)

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("📘 Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content!")

# Input Groq API key
api_key = st.text_input("Enter your Groq API key:", type="password")

# -----------------------------
# 4. Validate Groq API key
# -----------------------------
if not api_key:
    st.warning("Please enter your Groq API Key to continue.")
    st.stop()

# Initialize the LLM
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# -----------------------------
# 5. Session and History Management
# -----------------------------
session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

# -----------------------------
# 6. File Upload Section
# -----------------------------
uploaded_files = st.file_uploader("Choose one or more PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    temp_paths_to_cleanup=[]
    documents = []
    for uploaded_file in uploaded_files:
        temp_path = f"./temp_{uploaded_file.name}"
        if os.path.exists(temp_path):
            os.remove(temp_path)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

        temp_paths_to_cleanup.append(temp_path)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    st.session_state.retriever = vectorstore.as_retriever()

    for path in temp_paths_to_cleanup:
        if os.path.exists(path):
            os.remove(path)

    # -----------------------------
    # 7. Create Contextual Retriever
    # -----------------------------
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question — "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, st.session_state.retriever, contextualize_q_prompt)

    # -----------------------------
    # 8. Q&A System
    # -----------------------------
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # -----------------------------
    # 9. Session History Helper
    # -----------------------------
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # -----------------------------
    # 10. Chat Interaction
    # -----------------------------
    user_input = st.text_input("Ask a question about your PDFs:")

    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        st.write("**Assistant:**", response["answer"])
        st.write("**Chat History:**", session_history.messages)

