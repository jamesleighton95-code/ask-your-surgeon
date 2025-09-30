import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# -------------------------
# CONFIG
# -------------------------
RAW_FOLDER = "data/raw"
INDEX_FOLDER = "embeddings"


# -------------------------
# TEXT CLEANER
# -------------------------
import re

def clean_text(text: str) -> str:
    """Clean and normalise text, reformat BAUS-style risk tables into bullet points."""

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Detect BAUS risk table rows and convert to bullets
    # Matches: "Symptom ... Almost all patients", "Symptom ... 1 in 20 patients", "Symptom ... 5%" etc
    risk_pattern = re.compile(
        r"([A-Z][^%0-9]+?)\s+("
        r"(?:Almost all patients|Between [^%]+?|Less than [^%]+?|[0-9]+ in [0-9]+[^%]*|[0-9]+%|[0-9]+ ?\([0-9%]+\))"
        r")",
        flags=re.IGNORECASE
    )

    matches = risk_pattern.findall(text)
    if matches:
        bullets = [f"- {cond.strip()} → Risk: {risk.strip()}" for cond, risk in matches]
        return "\n".join(bullets)

    return text

# -------------------------
# MAIN INDEX BUILDER
# -------------------------
def build_index():
    embeddings = OpenAIEmbeddings()

    # Create output dir if missing
    os.makedirs(INDEX_FOLDER, exist_ok=True)

    # Walk through each condition folder (e.g. prostate, bph, bladder, kidney)
    for condition in os.listdir(RAW_FOLDER):
        cond_path = os.path.join(RAW_FOLDER, condition)
        if not os.path.isdir(cond_path):
            continue

        print(f"📑 Processing {condition}...")

        docs = []
        for fname in os.listdir(cond_path):
            if fname.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(cond_path, fname))
                pdf_docs = loader.load()

                # Clean text in each page
                for d in pdf_docs:
                    d.page_content = clean_text(d.page_content)
                    d.metadata["source"] = fname
                docs.extend(pdf_docs)

        if not docs:
            print(f"⚠️ No PDFs found in {cond_path}")
            continue



        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_docs = splitter.split_documents(docs)

        # ✅ Show first 500 characters so we can confirm numbers made it in
        sample_text = docs[0].page_content[:500]
        print(f"🔍 Sample from {condition}:")
        print(sample_text)

        # Build FAISS index
        db = FAISS.from_documents(split_docs, embeddings)

        # Save
        out_path = os.path.join(INDEX_FOLDER, f"{condition}_index")
        db.save_local(out_path)
        print(f"✅ Saved {condition} index at {out_path}")

if __name__ == "__main__":
    build_index()

