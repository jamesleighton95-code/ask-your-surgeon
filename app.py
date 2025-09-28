import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# Map conditions to multiple guideline sources
CONDITIONS = {
    "Prostate Cancer": [
        "data/clean/EAU_Prostate_Cancer_chunks.txt",
        "data/clean/Prostate_Cancer_UK_New_Diagnosis_chunks.txt",
        # Later you can add more:
        # "data/clean/NICE_Prostate_Cancer_chunks.txt",
        # "data/clean/BAUS_Prostate_Cancer_chunks.txt",
    ]
}

@st.cache_resource
def build_retriever(file_paths):
    """Builds a FAISS retriever by merging multiple text chunk files."""
    embeddings = OpenAIEmbeddings()
    docs = []

    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = f.readlines()
            docs.extend(
                [Document(page_content=chunk.strip(), metadata={"source": file_path})
                 for chunk in chunks if chunk.strip()]
            )
        except FileNotFoundError:
            st.warning(f"⚠️ File not found: {file_path}")

    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()

st.title("Ask Your Surgeon")
st.write("Understand your diagnosis, know your options, decide your treatment.")

# Patient selects their condition
condition_choice = st.selectbox("Select your condition:", list(CONDITIONS.keys()))

retriever = build_retriever(CONDITIONS[condition_choice])

query = st.text_input("Ask a question about your condition:")
if query:
    results = retriever.get_relevant_documents(query)
    st.subheader("Results")
    for r in results:
        st.write(r.page_content)
        st.caption(f"Source: {r.metadata.get('source', 'Unknown')}")

