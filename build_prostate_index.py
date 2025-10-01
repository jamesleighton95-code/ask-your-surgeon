# build_prostate_index.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Paths
RAW_DOCS = [
    "data/raw/EAU_Prostate_Guidelines.pdf",
    "data/raw/PCUK_New_Diagnosis.pdf"
]
INDEX_DIR = "embeddings/prostate_cancer_index"

# 1. Load PDFs
docs = []
for doc_path in RAW_DOCS:
    if os.path.exists(doc_path):
        loader = PyPDFLoader(doc_path)
        docs.extend(loader.load())
    else:
        print(f"⚠️ File not found: {doc_path}")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100
)
chunks = splitter.split_documents(docs)

# 3. Create embeddings
embeddings = OpenAIEmbeddings()

# 4. Build FAISS index
db = FAISS.from_documents(chunks, embeddings)

# 5. Save index
db.save_local(INDEX_DIR)

print(f"✅ Prostate cancer FAISS index rebuilt at: {INDEX_DIR}")

