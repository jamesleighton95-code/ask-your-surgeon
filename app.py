# app.py
import streamlit as st
from chatbot import qa_chain

st.set_page_config(page_title="Ask Your Surgeon", page_icon="💬")
st.title("💬 Ask Your Surgeon Chatbot")

st.write(
    "Ask questions about prostate cancer. "
    "Answers are simplified for patients and based on trusted guidelines (EAU, Prostate Cancer UK)."
)

query = st.text_input("Your question:")

if query:
    result = qa_chain(query)
    answer = result.get("result", "")
    sources = result.get("source_documents", [])

    st.markdown("### ✅ Answer")
    st.write(answer)

    if sources:
        st.markdown("### 📖 Sources")
        for doc in sources:
            st.write(f"- {doc.metadata.get('source', 'Unknown source')}")

