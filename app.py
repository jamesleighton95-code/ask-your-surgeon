import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from master_slave_validator import load_specialties, ask_chatbot, MASTER_TEMPLATE

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="Ask Your Surgeon – Urology Chatbot", layout="wide")

st.title("💬 Ask Your Surgeon – Urology Chatbot")
st.write("This chatbot provides information based on trusted sources (EAU, BAUS, PCUK).")

# Cache the bot so it doesn't reload every time
@st.cache_resource
def load_bot():
    slaves = load_specialties()  # ✅ defaults to "embeddings/"
    master_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=PromptTemplate.from_template(MASTER_TEMPLATE)
    )
    return slaves, master_chain

slaves, master_chain = load_bot()

# Input box
user_query = st.text_input("Ask me about your urological condition:")

if user_query:
    with st.spinner("Consultant is thinking..."):
        answer = ask_chatbot(
            user_query,
            slaves,
            master_chain,
            debug=True  # set False to hide debug info
        )
    st.markdown(answer)

    # Optional: Debug info (toggle on demand)
    with st.expander("🔎 Debug info"):
        st.write("This section shows retrieved chunks, verdicts, and raw outputs for debugging.")
        # In the current implementation, ask_chatbot already prints debug info to console.
        # If you want to capture and show it here, you'd need to adjust ask_chatbot to return it.

