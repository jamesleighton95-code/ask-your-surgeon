import os
import re
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader


# -------------------------
# CONFIG
# -------------------------
RAW_FOLDER = "data/raw"
INDEX_FOLDER = "embeddings"

# -------------------------
# TEXT CLEANER
# -------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    risk_like_pattern = re.compile(
        r"(?:\d+ in \d+|less than \d+ in \d+|fewer than \d+ in \d+|up to \d+ in \d+|"
        r"between \d+ and \d+ in \d+|\d+%|\d+-\d+%|"
        r"almost all(?: patients)?|most(?: patients)?|majority(?: of patients)?)",
        flags=re.IGNORECASE
    )
    return risk_like_pattern.sub("[RISK]", text)  # replaces instead of dropping whole page

# -------------------------
# MAIN INDEX BUILDER
# -------------------------
def build_index():
    embeddings = OpenAIEmbeddings()
    os.makedirs(INDEX_FOLDER, exist_ok=True)

    # Walk through each condition folder
    for condition in os.listdir(RAW_FOLDER):
        cond_path = os.path.join(RAW_FOLDER, condition)
        if not os.path.isdir(cond_path):
            continue

        pdf_files = [f for f in os.listdir(cond_path) if f.endswith(".pdf")]
        total_pdfs = len(pdf_files)

        print(f"\n📑 Processing {condition} ({total_pdfs} PDFs)...")

        docs = []

        for i, fname in enumerate(pdf_files, start=1):
            file_path = os.path.join(cond_path, fname)
            print(f"   [{i}/{total_pdfs}] Loading {fname}...")

            try:
                if "EAU" in fname.upper():
                    # 🔑 Use PyMuPDF for EAU guidelines (better text extraction)
                    loader = PyMuPDFLoader(file_path)
                    pdf_docs = loader.load()
                    print(f"   ✅ Parsed {fname} with PyMuPDF (EAU guideline)")
                else:
                    # 🔑 Use Unstructured for BAUS / NICE / PCUK
                    loader = UnstructuredPDFLoader(file_path, strategy="fast")
                    pdf_docs = loader.load()
                    print(f"   ✅ Parsed {fname} with fast mode")

                    # ⚠️ If nothing came out, retry with OCR
                    if not any(d.page_content.strip() for d in pdf_docs):
                        print(f"   ⚠️ No text extracted from {fname} with fast mode, retrying with OCR...")
                        loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
                        pdf_docs = loader.load()
                        print(f"   ✅ Parsed {fname} with OCR fallback")

            except Exception as e:
                print(f"   ❌ Failed to parse {fname}: {e}")
                continue

            # Process text depending on file type
            kept = 0
            for d in pdf_docs:



                d.page_content = d.page_content.strip()



                d.metadata["source"] = fname
                docs.append(d)
                kept += 1

            print(f"   📌 Kept {kept} chunks from {fname}")

        if not docs:
            print(f"⚠️ No documents for {condition}, skipping.")
            continue



            try:
                if "EAU" in fname.upper():
                    # 🔑 Use PyMuPDF for EAU guidelines (better text extraction)
                    loader = PyMuPDFLoader(file_path)
                    pdf_docs = loader.load()
                    print(f"   ✅ Parsed {fname} with PyMuPDF (EAU guideline)")
                else:
                    # 🔑 Use Unstructured for BAUS / NICE / PCUK
                    loader = UnstructuredPDFLoader(file_path, strategy="fast")
                    pdf_docs = loader.load()
                    print(f"   ✅ Parsed {fname} with fast mode")

                    # ⚠️ If nothing came out, retry with OCR
                    if not any(d.page_content.strip() for d in pdf_docs):
                        print(f"   ⚠️ No text extracted from {fname} with fast mode, retrying with OCR...")
                        loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
                        pdf_docs = loader.load()
                        print(f"   ✅ Parsed {fname} with OCR fallback")

            except Exception as e:
                print(f"   ❌ Failed to parse {fname}: {e}")
                continue

            # Process text depending on file type
            kept = 0
            for d in pdf_docs:
                if "BAUS" in fname.upper():
                    # ✅ Clean only BAUS leaflets (strip risk statistics)
                    cleaned = clean_text(d.page_content)
                    if not cleaned.strip():
                        print(f"      ⚠️ Dropped a chunk from {fname}: {d.page_content[:200]}...")
                        continue
                    d.page_content = cleaned
                else:
                    # ✅ Keep full text for EAU / NICE / PCUK etc.
                    d.page_content = d.page_content.strip()

                d.metadata["source"] = fname
                docs.append(d)
                kept += 1

            print(f"   📌 Kept {kept} chunks from {fname}")

        if not docs:
            print(f"⚠️ No documents for {condition}, skipping.")
            continue

        # 🔍 Debug sample
        sample_text = docs[0].page_content[:500]
        print(f"🔍 Sample from {condition}:\n{sample_text}\n")

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # ✅ safer size (avoid token overload)
            chunk_overlap=150
        )
        split_docs = splitter.split_documents(docs)

        # Build FAISS index
        db = FAISS.from_documents(split_docs, embeddings)

        # Save index
        out_path = os.path.join(INDEX_FOLDER, f"{condition}_index")
        db.save_local(out_path)
        print(f"✅ Saved {condition} index at {out_path} ({len(split_docs)} chunks)")


if __name__ == "__main__":
    build_index()

