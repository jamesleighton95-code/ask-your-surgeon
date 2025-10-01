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
from postprocessors import enforce_exact_risks
from aliases import BAUS_DOC_MAP, GROUND_TRUTH_MAP

# ✅ Force debug mode on
debug = True

# -------------------------
# ✅ Ground truth detection
# -------------------------
RISK_TERMS = [
    "risk", "risks", "complication", "complications",
    "side effect", "side effects", "problem", "problems"
]

def get_ground_truth_file(query: str):
    """Return the path to a ground truth file if query matches."""
    q_lower = query.lower()
    for key, filepath in GROUND_TRUTH_MAP.items():
        if key in q_lower and any(term in q_lower for term in RISK_TERMS):
            return filepath
    return None



# ✅ Force debug mode on
debug = True


# ----------------------
# Conversation Memory
# ----------------------
chat_history = []


# ----------------------
# Prompt Templates
# ----------------------


MASTER_TEMPLATE = """You are a senior NHS Urology Consultant.
Your job is to:
1. Rewrite the patient’s latest question so that it is self-contained.
2. Decide which specialty area (e.g. prostate, bladder, kidney, BPH, stones) the rewritten query belongs to.

Rules:  
- If the question is about prostate enlargement, LUTS, flow, peeing at night, or BPH medications (tamsulosin/finasteride/etc.), choose "bph" - NOT prostate cancer.
- If it is about prostate cancer (diagnosis, staging, treatment), choose "prostate".
- Only choose from this list: {allowed_specialties}.
- If nothing fits, set "specialty" to "unsupported".

Conversation so far:
{question}

Return JSON in this exact format:
{{"query": "<rewritten self-contained question>", "specialty": "<specialty>"}}
"""

CONSULTANT_TEMPLATE = """You are a senior Urology Consultant working in the NHS.
Your role is to help patients who have already been diagnosed with a urological condition
fully understand their condition, explore treatment options, and prepare to give informed consent.

When answering:
- If the question mentions cancer, complications, or prognosis → begin with a short empathetic statement (e.g., "I understand this can feel worrying" / "Many people share this concern"). 
- If the question is a neutral “what is…” or factual query → begin directly with a clear, simple explanation without extra empathy. 
- Always use patient-friendly language.
- Then move onto an initial 1-2 focused answer in bullet point format

Strict rules:
- Only use information from trusted sources (EAU guidelines, BAUS resources, Prostate Cancer UK, NHS leaflets).
- You must never invent percentages, statistics, or risks. 
- Never generalise or guess.
- Try not to give specific risk fractions or percentages, keep it vague until the patient asks specifically about risks
- Do not paraphrase or simplify numerical risks.  


Professionalism & Accessibility:
- Always communicate empathetically and respectfully, in plain English.
- Avoid unnecessary medical jargon; explain terms simply when needed.
- Make content accessible for patients of all backgrounds and literacy levels.
- Align responses with NHS practice, BAUS guidance, and EAU evidence-based recommendations.

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
 
    # ✅ Handle vague / unsupported routing gracefully
    if routed_spec == "unsupported":
        return (
            "Consultant: I want to make sure I give you the most accurate information. "
            "Could you please be a little more specific about what you’d like me to explain — "
            "for example, the risks of TURP, HoLEP, or another procedure?"
        )

    if routed_spec not in slaves:
        return f"⚠️ Sorry, I can’t handle that topic. Available: {', '.join(slaves.keys())}"

    result = slaves[routed_spec]({"question": rewritten_q, "query": rewritten_q})
    answer = result["result"]
    docs = result.get("source_documents", [])

    RISK_TERMS = [
        "risk", "risks", "complication", "complications",
        "side effect", "side effects", "problem", "problems"
    ]

    lowered_q = rewritten_q.lower()
    baus_key = next((k for k in BAUS_DOC_MAP if k in lowered_q), None)

    # 1. Ground truth override
    if baus_key and baus_key in GROUND_TRUTH_MAP:
        if any(term in lowered_q for term in RISK_TERMS):
            gt_path = GROUND_TRUTH_MAP[baus_key]
            with open(gt_path, "r") as f:
                final_answer = f.read().strip()
            if debug:
                print(f"[DEBUG] Using ground truth risks for {baus_key.upper()}")
            return (
                f"Consultant: Here are the official risks from BAUS "
                f"({baus_key.upper()} leaflet):\n\n{final_answer}"
            )

    if baus_key and not any(term in lowered_q for term in RISK_TERMS):
        # Allow retrieval from all trusted docs, not just BAUS
        retriever = slaves[routed_spec].retriever
        results = retriever.get_relevant_documents(rewritten_q)

        if results:
            docs = results
            if debug:
                print(f"[DEBUG] Non-risk BAUS query → using retrieval across trusted docs ({len(docs)} chunks)")

        # Hide specific figures in non-risk queries
        final_answer = enforce_exact_risks(answer, [d.page_content for d in docs], debug=debug)
        final_answer = re.sub(r"\b\d+(\s*\/\s*\d+)?\s*%|\b\d+\s+in\s+\d+\b", "[risk figures removed]", final_answer)

        return (
            f"Consultant: {final_answer}\n\n"
            "This information is general and based on trusted sources "
            "(BAUS, EAU, NICE, NHS). It is not a substitute for personal advice "
            "from your own doctor."
        )

    # 3. Fallback → generic sources
    retrieved_texts = [d.page_content for d in docs]
    final_answer = enforce_exact_risks(answer, retrieved_texts, debug=debug)

    if not any(term in lowered_q for term in RISK_TERMS):
        if debug:
            print("[DEBUG] Non-risk query → hiding numerical risk figures from fallback sources")
        final_answer = re.sub(
            r"\b\d+(\s*\/\s*\d+)?\s*%|\b\d+\s+in\s+\d+\b",
            "[risk figures removed]",
            final_answer
        )

    # Save exchange into chat history
    chat_history.append((user_query, final_answer))

    sources = [os.path.basename(d.metadata.get("source", "?")) for d in docs]
    unique_sources = sorted(set(sources))
    sources_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(unique_sources))

    return f"{final_answer}\n\n📖 Sources used:\n{sources_str}"

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

