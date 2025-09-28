import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain

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

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

@st.cache_resource
def load_chat_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Patient-friendly prompt
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful urology consultant.\n"
            "Answer the patient's question clearly and kindly, using simple, patient-friendly language.\n"
            "Avoid jargon unless absolutely necessary. If you use a medical term, explain it.\n"
            "Base your answer only on the following trusted guideline context:\n\n"
            "{context}\n\n"
            "Patient's question: {question}\n\n"
            "Final answer (friendly, clear, concise):"
        )
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )


st.title("Ask Your Surgeon")
st.write("Understand your diagnosis, know your options, decide your treatment.")

# Patient selects condition
condition_choice = st.selectbox("Select your condition:", list(CONDITIONS.keys()))

retriever = build_retriever(CONDITIONS[condition_choice])
chat_chain = load_chat_chain(retriever)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask your question here...")
if query:
    # Append user query
    st.session_state.messages.append({"role": "user", "content": query})

    # Run chain with conversation history
    response = chat_chain.invoke({
        "question": query,
        "chat_history": [(m["role"], m["content"]) for m in st.session_state.messages if m["role"] != "assistant"]
    })

    # Append assistant response
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Show sources for the latest answer
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    st.subheader("Sources")
    for doc in response["source_documents"]:
        st.caption(f"- {doc.metadata.get('source', 'Unknown')}")

