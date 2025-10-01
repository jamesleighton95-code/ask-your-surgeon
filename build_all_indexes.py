# build_all_indexes.py
import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

RAW_DIR = "data/raw"
EMBED_DIR = "embeddings"

def build_index(condition: str, folder: str, out_dir: str = EMBED_DIR):
    """Build a FAISS index for a condition from all PDFs in the given folder."""

    # Find all PDFs inside the folder
    pdf_files = glob(os.path.join(folder, "*.pdf"))
    if not pdf_files:
        print(f"⚠️ No PDFs found in {folder}, skipping...")
        return

    docs = []
    for file_path in pdf_files:
        print(f"📄 Loading: {file_path}")
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ {condition}: split into {len(chunks)} chunks")

    # Embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS index
    db = FAISS.from_documents(chunks, embeddings)

    # Save to condition-specific directory
    index_dir = os.path.join(out_dir, f"{condition.lower()}_index")
    db.save_local(index_dir)

    print(f"✅ {condition.capitalize()} index saved at: {index_dir}\n")

def build_all_indexes():
    """Scan data/raw for subfolders and build an index for each."""
    if not os.path.exists(RAW_DIR):
        raise ValueError(f"❌ No {RAW_DIR} folder found")

    subfolders = [f.path for f in os.scandir(RAW_DIR) if f.is_dir()]

    if not subfolders:
        raise ValueError("❌ No subfolders found in data/raw/. Please add condition folders.")

    for folder in subfolders:
        condition = os.path.basename(folder)
        build_index(condition, folder)

if __name__ == "__main__":
    build_all_indexes()

