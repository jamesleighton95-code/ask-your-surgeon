import os
import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- Config ----------
FAISS_PATH = "embeddings/eau_prostate_cancer_index"
EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"
TOP_K = 3
MAX_TOKENS = 600
TEMPERATURE = 0.2
# ----------------------------

# Load FAISS retriever
@st.cache_resource
def load_retriever():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db.as_retriever(search_kwargs={"k": TOP_K})

retriever = load_retriever()
client = OpenAI()

st.set_page_config(page_title="Ask Your Surgeon", page_icon="🩺")
st.title("🩺 Ask Your Surgeon — Urology Chatbot")
st.markdown("**Disclaimer:** This chatbot is for educational purposes only and is *not* a substitute for professional medical advice. Always consult a qualified urologist.")

# Chat input
query = st.text_input("Enter your question:")

if query:
    # Retrieve relevant docs
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([d.page_content for d in docs])

    # Build prompts
    system_prompt = (
        "You are a careful medical assistant specializing in urology. "
        "Answer ONLY using the provided CONTEXT. If the answer is not in the context, say "
        "\"I don't have enough information from the provided sources.\" "
        "Keep answers concise, patient-friendly, and avoid speculation. "
        "ALWAYS include this disclaimer at the end: "
        "\"This is educational only and not a substitute for medical advice. "
        "Consult a urologist for diagnosis and treatment.\""
    )

    user_prompt = f"CONTEXT:\n{context_text}\n\nPATIENT QUESTION:\n{query}"

    # Generate answer
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = resp.choices[0].message.content

    st.subheader("🩺 Answer")
    st.write(answer)

    with st.expander("📚 Sources"):
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Source {i}:** {d.page_content[:400]}...")


