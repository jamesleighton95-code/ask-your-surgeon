from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# Path to your Prostate Cancer index
INDEX_PATH = "embeddings/prostate_cancer_index"

# Load the FAISS index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# LLM for generating answers
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def query(question):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([f"[{d.metadata['source']}]\n{d.page_content}" for d in docs])

    prompt = f"""You are a helpful assistant for patients with prostate cancer.
Use the context below to answer the question.
If unsure, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

    response = llm.invoke(prompt)
    return response.content, docs

if __name__ == "__main__":
    while True:
        q = input("\n❓ Ask a question (or type 'exit'): ")
        if q.lower() in ["exit", "quit"]:
            break
        answer, docs = query(q)
        print("\n💡 Answer:", answer)
        print("\n📖 Sources used:")
        for d in docs:
            print("-", d.metadata["source"])

