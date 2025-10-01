from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("embeddings/prostate_index", embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 3})
docs = retriever.get_relevant_documents("what is prostate cancer")

for i, d in enumerate(docs, 1):
    print(f"[{i}] {d.page_content[:200]}...")

