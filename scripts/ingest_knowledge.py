
import os
import sys
import shutil
from typing import List

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
KNOWLEDGE_DIR = "knowledge"
DB_DIR = "knowledge_base"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_docs():
    # 1. Load Documents
    print(f"Loading documents from {KNOWLEDGE_DIR}...")
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        print("Created knowledge directory.")
        return

    loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.md", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print("No documents found to ingest.")
        return

    # 2. Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Create Vector Store
    # Clear existing DB if needed (optional, here we append or overwrite?)
    # For simplicity, let's clear and rebuild to avoid duplicates in this demo
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("Cleared existing database.")

    print("Creating vector store...")
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=DB_DIR
    )
    print(f"Vector store created in {DB_DIR}")

if __name__ == "__main__":
    ingest_docs()
