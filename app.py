import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

# Map conditions to multiple guideline sources
CONDITIONS = {
    "Prostate Cancer": [
        "data/clean/EAU_Prostate_Cancer_chunks.txt",
        "data/clean/Prostate_Cancer_UK_New_Diagnosis_chunks.txt",
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

@st.cache_resource
def load_qa_chain(retriever):
    """Builds a RetrievalQA chain using GPT and the retriever."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

st.title("Ask Your Surgeon")
st.write("Understand your diagnosis, know your options, decide your treatment.")

# Patient selects condition
condition_choice = st.selectbox("Select your condition:", list(CONDITIONS.keys()))

retriever = build_retriever(CONDITIONS[condition_choice])
qa_chain = load_qa_chain(retriever)

query = st.text_input("Ask a question about your condition:")
if query:
    response = qa_chain.invoke({"query": query})
    
    st.subheader("Answer")
    st.write(response["result"])

    st.subheader("Sources")
    for doc in response["source_documents"]:
        st.caption(f"- {doc.metadata.get('source', 'Unknown')}")

