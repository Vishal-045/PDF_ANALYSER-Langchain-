# pdf_qa_app.py

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# ------------------------------------
# Streamlit Setup
# ------------------------------------
st.set_page_config(page_title="📘 PDF Q&A with LangChain FAISS", layout="wide")
st.title("📘 AI-Powered PDF Analyzer (LangChain)")
st.write("Upload a PDF and ask any question about its content using LangChain + Transformers.")

# ------------------------------------
# File Upload
# ------------------------------------
uploaded_file = st.file_uploader("📂 Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    st.info("📄 Extracting text from the uploaded PDF...")

    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text + "\n"

    if not pdf_text.strip():
        st.error("❌ No readable text found in the PDF.")
        st.stop()

    # Split into chunks
    st.info("🪓 Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(pdf_text)
    st.success(f"✅ Extracted and split into {len(chunks)} chunks.")

    # ------------------------------------
    # Build Vectorstore with LangChain
    # ------------------------------------
    st.info("🔍 Creating FAISS vectorstore with HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    st.success("✅ Vectorstore created successfully!")

    # ------------------------------------
    # Load QA Model
    # ------------------------------------
    st.info("🤖 Loading FLAN-T5-small model for answering questions...")
    generator = pipeline("text2text-generation", model="google/flan-t5-small")

    # ------------------------------------
    # Query Section
    # ------------------------------------
    st.markdown("---")
    st.subheader("💬 Ask a question about your PDF content:")
    query = st.text_input("Enter your question here:")

    if query:
        with st.spinner("🔍 Retrieving relevant context and generating answer..."):
            # Retrieve top 3 similar chunks
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Prepare prompt
            prompt = f"""
You are an AI assistant that answers questions based on PDF content.
Use the provided context to answer accurately and concisely.
If the answer isn't in the context, say: "I don't have that information."

Context:
{context}

Question:
{query}

Answer:
"""
            # Generate answer
            response = generator(prompt, max_length=256, do_sample=False)
            answer = response[0]['generated_text']

        # Display results
        st.markdown("### 🧠 Answer:")
        st.write(answer)

        with st.expander("📘 Show Retrieved Context"):
            st.write(context)

else:
    st.info("👆 Please upload a PDF file to get started.")
