PDF Q&A App is an AI-powered application that allows users to upload any PDF and ask questions about its content.
The system intelligently extracts text, creates vector embeddings using HuggingFace models, stores them in a FAISS vector database, retrieves relevant chunks, and generates an answer using FLAN-T5.

This project demonstrates modern Retrieval-Augmented Generation (RAG) using LangChain and Transformers, wrapped in a clean Streamlit UI.# PDF_ANALYSER
⭐ Features

Upload and process any text-based PDF

Automatic text extraction using PyPDF2

Smart text chunking with RecursiveCharacterTextSplitter

Vector embeddings using all-MiniLM-L6-v2

FAISS-based similarity search

Answer generation using FLAN-T5-small

Clean and interactive Streamlit interface

Context inspection for transparency

🛠️ Tech Stack

Python 3.8+

Streamlit – UI framework

LangChain – embeddings, vectorstore

FAISS – vector search

HuggingFace Transformers – FLAN-T5 model

PyPDF2 – PDF text extraction
