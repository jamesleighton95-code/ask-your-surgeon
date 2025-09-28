import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# Path to your cleaned chunks file
CHUNKS_FILE = "data/clean/Prostate_Cancer_UK_New_Diagnosis_chunks.txt"

@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings()

    # Load chunks from txt file instead of FAISS binaries
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = f.readlines()

    docs = [Document(page_content=chunk.strip()) for chunk in chunks if chunk.strip()]
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()

retriever = load_retriever()

st.title("Ask Your Surgeon")
st.write("Understand your diagnosis, know your options, decide your treatment.")

query = st.text_input("Ask a question about Prostate Cancer:")
if query:
    results = retriever.get_relevant_documents(query)
    for r in results:
        st.write(r.page_content)

