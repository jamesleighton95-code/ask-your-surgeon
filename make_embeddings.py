from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os

# Input file with chunked guideline
input_file = "data/clean/EAU_Prostate_Cancer_chunks.txt"

# Read chunks
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

chunks = content.split("--- Chunk ")
docs = []
for chunk in chunks:
    if chunk.strip():
        docs.append(Document(page_content=chunk.strip(), metadata={"source": "EAU_Prostate_Cancer"}))

# Create embeddings object
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed in safe batches of 50 chunks
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

# Save FAISS index
os.makedirs("embeddings", exist_ok=True)
db.save_local("embeddings/eau_prostate_cancer_index")

print(f"✅ Created FAISS vector database with {len(docs)} chunks")

