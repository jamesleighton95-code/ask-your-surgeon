from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load the saved FAISS DB
db = FAISS.load_local(
    "embeddings/eau_prostate_cancer_index",
    OpenAIEmbeddings(model="text-embedding-3-small"),
    allow_dangerous_deserialization=True
)

# Example query
query = "What are the treatment options for prostate cancer?"
results = db.similarity_search(query, k=3)

print(f"\n🔎 Query: {query}\n")
for i, res in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(res.page_content[:500])  # print first 500 characters of the chunk
    print()


