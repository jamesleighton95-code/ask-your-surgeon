# build_index.py
import os
import argparse
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_index(condition: str, folder: str, out_dir: str = "embeddings"):
    """Build a FAISS index for a condition from all PDFs in the given folder."""

    # Find all PDFs inside the folder
    pdf_files = glob(os.path.join(folder, "*.pdf"))
    if not pdf_files:
        raise ValueError(f"❌ No PDF files found in {folder}")

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
    print(f"✅ Split into {len(chunks)} chunks")

    # Embeddings
    embeddings = OpenAIEmbeddings()

    # Create FAISS index
    db = FAISS.from_documents(chunks, embeddings)

    # Save to condition-specific directory
    index_dir = os.path.join(out_dir, f"{condition.lower()}_index")
    db.save_local(index_dir)

    print(f"✅ {condition.capitalize()} FAISS index built at: {index_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS index for a urology condition")
    parser.add_argument("--condition", type=str, required=True, help="Condition name (e.g. prostate, bladder)")
    parser.add_argument("--folder", type=str, required=True, help="Folder containing PDFs for this condition")
    args = parser.parse_args()

    build_index(args.condition, args.folder)

