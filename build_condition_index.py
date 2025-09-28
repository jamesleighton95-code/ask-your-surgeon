import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def build_condition_index(files, output_folder, condition_name):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = []

    for fpath in files:
        if not os.path.exists(fpath):
            print(f"⚠️ Skipping missing file: {fpath}")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = content.split("--- Chunk ")
        for chunk in chunks:
            if chunk.strip():
                docs.append(Document(
                    page_content=chunk.strip(),
                    metadata={"condition": condition_name, "source": os.path.basename(fpath)}
                ))

    if not docs:
        print(f"❌ No documents found for {condition_name}")
        return

    batch_size = 50
    db = None
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        print(f"🔄 Processing batch {i//batch_size + 1} ({len(batch)} docs)")
        batch_db = FAISS.from_documents(batch, embeddings)
        if db is None:
            db = batch_db
        else:
            db.merge_from(batch_db)

    os.makedirs(output_folder, exist_ok=True)
    db.save_local(output_folder)
    print(f"✅ Built FAISS index for {condition_name} with {len(docs)} chunks")

if __name__ == "__main__":
    build_condition_index(
        [
            "data/clean/EAU_Prostate_Cancer_chunks.txt",
            "data/clean/Prostate_Cancer_UK_New_Diagnosis_chunks.txt",
        ],
        "embeddings/prostate_cancer_index",
        "Prostate Cancer"
    )

