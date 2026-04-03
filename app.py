import streamlit as st
import tempfile

# Document loading & splitting
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# HuggingFace model (text generation)
from transformers import pipeline

st.title("📄 FREE RAG Chatbot (HuggingFace)")

# Upload file
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Load document
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings()

    # Store in FAISS
    db = FAISS.from_documents(docs, embeddings)

    st.success("✅ Document processed!")

    # Ask question
    query = st.text_input("Ask your question:")

    if query:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(query)

        # Combine context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

        # HuggingFace text-generation pipeline
        text_gen = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_length=512
        )

        result = text_gen(prompt, do_sample=False)

        st.subheader("🤖 Answer:")
        st.write(result[0]["generated_text"])
