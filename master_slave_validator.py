from datetime import datetime
import os
import re
import json
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ----------------------
# Conversation Memory
# ----------------------
chat_history = []


# ----------------------
# Prompt Templates
# ----------------------

MASTER_TEMPLATE = """You are a senior NHS Urology Consultant.
Your job is to decide which specialty area (e.g. prostate, bladder, kidney, BPH, stones) a patient's question belongs to.
You are not answering the question yourself - only rewriting it into a precise clinical query and assigning it to the correct specialty.

Rules:
- If the question is about prostate enlargement, LUTS, flow, peeing at night, or BPH medications (tamsulosin/finasteride/etc.), choose "bph" - NOT prostate cancer.
- If it is about prostate cancer (diagnosis, staging, treatment), choose "prostate".
- Only choose from this list: {allowed_specialties}.
- If nothing fits, set "specialty" to "unsupported".

Conversation so far:
{question}

Decide the specialty for the patient’s **latest question**, using the conversation history above if needed.

Return ONLY one JSON line:
{{"query": "<rewritten clinical query>", "specialty": "<one of: {allowed_specialties}, unsupported>"}}
"""

CONSULTANT_TEMPLATE = """You are a senior Urology Consultant working in the NHS.
Your role is to help patients who have already been diagnosed with a urological condition
fully understand their condition, explore treatment options, and prepare to give informed consent.

Always follow this structure:

1. Short Summary
   - Provide a simple 2–3 sentence explanation in plain English.
   - Avoid medical jargon. Be clear, empathetic, and patient-friendly.

2. Offer Expansion
   - Add new detail not already covered in the Short Summary.
   - Do NOT repeat or rephrase content from the Short Summary.
   - Expand with staging, treatment types, risks, and consent points where relevant.
   - End with: "I can provide more detailed information on treatments, risks, or consent if you’d like."
3. Confirm Understanding
   - End with one **simple reflective check**, e.g.:
     "Does this explanation make sense so far?"
     or
     "Would you like me to go over the risks in more detail?"
   - Always **end by asking the patient** if the explanation makes sense so far.
   - Do **NOT** invent or write patient responses. Wait for the patient to answer in the next turn.


When asked for *more detail*, expand into:
   - **Diagnosis explanation**: what the condition is, how it affects health.
   - **Treatment options**: list surgical and non-surgical approaches, each with pros/cons and risks.
   - **Consent preparation**: summarise key points a patient must know before treatment.

Strict rules:
- Only use information from trusted sources (EAU guidelines, BAUS resources, Prostate Cancer UK, NHS leaflets).
- You must never invent percentages, statistics, or risks. 
- Only state numbers if they are explicitly written in the context.
- If the patient specifically asks for percentages or numbers and they are not in the context, reply:
  "⚠️ I’m sorry, I cannot provide percentages for that risk because they are not available in my trusted resources."
- Never generalise or guess.

Professionalism & Accessibility:
- Always communicate empathetically and respectfully, in plain English.
- Avoid unnecessary medical jargon; explain terms simply when needed.
- Make content accessible for patients of all backgrounds and literacy levels.
- Align responses with NHS practice, BAUS guidance, and EAU evidence-based recommendations.

You must ONLY use the information in the context below.
If you cannot find an answer directly in the context, reply: "⚠️ I'm sorry, I don't have that information in my resources."

Context:
{context}

Patient question: {question}
Consultant:
"""

# ----------------------
# Semantic Validator
# ----------------------

validator_model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_validator(answer: str, docs, threshold: float = 0.65) -> bool:
    """Check if the consultant's answer is supported by retrieved context."""

    # Extract context
    context = " ".join([d.page_content for d in docs])

    # Split into sentences
    sentences = re.split(r"(?<=[.!?]) +", answer)

    # Define trivial patterns to skip
    trivial_patterns = [
        r"^I can provide more detailed information",
        r"^I can give you more detailed information",
        r"^Your doctor",
        r"^It is important",
        r"^I’m sorry",
        r"^⚠️",
    ]

    # Filter sentences
    filtered_sentences = [
        s for s in sentences
        if s.strip() and not any(re.match(p, s.strip()) for p in trivial_patterns)
    ]

    if not filtered_sentences:
        return False

    # Embeddings
    embeddings = OpenAIEmbeddings()
    context_emb = embeddings.embed_query(context)

    supported = 0
    for s in filtered_sentences:
        sent_emb = embeddings.embed_query(s)
        score = cosine_similarity(
            [sent_emb],
            [context_emb]
        )[0][0]
        if score >= threshold:
            supported += 1

    support_ratio = supported / len(filtered_sentences)
    print(f"[DEBUG] Supported {supported}/{len(filtered_sentences)} sentences ({support_ratio:.0%})")

    return support_ratio >= 0.5  # passes if ≥ 50% supported


# ----------------------
# Utility Functions
# ----------------------

def extract_json(text):
    """Extract first JSON object from model output safely."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


def log_interaction(user_query, routed_spec, docs, answer):
    """Save each Q&A to a log file for audit."""
    with open("chat_log.txt", "a") as f:
        f.write(f"--- {datetime.now()} ---\n")
        f.write(f"Q ({routed_spec}): {user_query}\n")
        f.write(f"A: {answer}\n")
        f.write(f"Sources: {[os.path.basename(d.metadata.get('source','?')) for d in docs]}\n\n")


# ----------------------
# Load Specialties
# ----------------------

def load_specialties(base_path="embeddings/"):
    """Load all FAISS indexes for each specialty."""
    embeddings = OpenAIEmbeddings()
    specialties = {}
    print(f"🔎 Loading specialties from '{base_path}'...")

    for folder in os.listdir(base_path):
        path = os.path.join(base_path, folder)
        if os.path.isdir(path):
            try:
                db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
                retriever = db.as_retriever(search_kwargs={"k": 6})

                chain = RetrievalQA.from_chain_type(
                    llm=OpenAI(temperature=0, max_tokens=800),
                    retriever=retriever,
                    chain_type="stuff",
                    chain_type_kwargs={
                        "prompt": PromptTemplate(
                            template=CONSULTANT_TEMPLATE,
                            input_variables=["context", "question"]
                        ),
                        "document_variable_name": "context",  # 👈 ensures docs are passed correctly
                    },
                    return_source_documents=True,  # 👈 ensures sources come back
                )

                specialties[folder] = chain
            except Exception as e:
                print(f"⚠️ Failed to load {folder}: {e}")

    print(f"✅ Loaded specialties: {', '.join(specialties.keys())}")
    return specialties


# ----------------------
# Ask Chatbot
# ----------------------

# Keep chat history globally or pass it in
chat_history = []

def ask_chatbot(user_query, slaves, master_chain, debug=False):
    global chat_history

    # Allow user to reset history
    if user_query.lower() in ["reset", "clear", "start again"]:
        chat_history = []
        return "🧹 Chat history cleared. Please ask your first question."


    # Include chat history in the master routing
    full_query = "\n".join(
        [f"Patient: {q}\nConsultant: {a}" for q, a in chat_history[-3:]]
    ) + f"\nPatient: {user_query}"

    if debug:
        print("\n[DEBUG] Full query sent to master router:\n", full_query)

    master_prompt = master_chain.run({
        "question": full_query,
        "allowed_specialties": ", ".join(sorted(slaves.keys()))
    })

    if debug:
        print("\n[DEBUG] Master router raw output:\n", master_prompt)

    try:
        parsed = json.loads(master_prompt.strip())
        rewritten_q = parsed["query"]
        routed_spec = parsed["specialty"]
    except Exception as e:
        return f"❌ Routing failed: {e}"

    if routed_spec not in slaves:
        return f"⚠️ Sorry, I can’t handle that topic. Available: {', '.join(slaves.keys())}"

    result = slaves[routed_spec]({"question": rewritten_q, "query": rewritten_q})
    answer = result["result"]
    docs = result.get("source_documents", [])

    # Run semantic validator
    if not semantic_validator(answer, docs):
        return "⚠️ I'm sorry, I can't fully support that answer from my trusted resources."

    # Save exchange into chat history
    chat_history.append((user_query, answer))

    sources = [os.path.basename(d.metadata.get("source", "?")) for d in docs]
    unique_sources = sorted(set(sources))
    sources_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(unique_sources))

    return f"{answer}\n\n📖 Sources used:\n{sources_str}"




def remove_duplicates(text: str) -> str:
    """Remove duplicate sentences while preserving order."""
    seen = set()
    cleaned = []
    for sentence in re.split(r'(?<=[.!?]) +', text):
        s = sentence.strip()
        if s and s not in seen:
            cleaned.append(s)
            seen.add(s)
    return " ".join(cleaned)



# ----------------------
# Main (for terminal use)
# ----------------------

def main():
    slaves = load_specialties("embeddings/")
    master_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=PromptTemplate.from_template(MASTER_TEMPLATE)
    )

    print("💬 Master–Slave–Validator Chatbot (type 'quit' to exit)\n")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["quit", "exit"]:
            break
        answer = ask_chatbot(user_query, slaves, master_chain, debug=True)
        print(f"\nConsultant: {answer}\n")


if __name__ == "__main__":
    main()

