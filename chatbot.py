import os
import textwrap
from typing import List
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- Config ----------
FAISS_PATH = "embeddings/eau_prostate_cancer_index"
EMBED_MODEL = "text-embedding-3-small"   # embeddings
GEN_MODEL   = "gpt-4o-mini"              # generator (cheap & good)
TOP_K = 3                                # how many chunks to retrieve
MAX_TOKENS = 600                         # max tokens for the answer
TEMPERATURE = 0.2                        # keep it factual
# ----------------------------

def load_retriever():
    # Load FAISS index with the same embedding model used to build it
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    db = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

def make_system_prompt() -> str:
    return (
        "You are a careful medical assistant specializing in urology. "
        "Answer ONLY using the provided CONTEXT. If the answer is not in the context, say "
        "\"I don't have enough information from the provided sources.\" "
        "Keep answers concise, patient-friendly, and avoid speculation. "
        "ALWAYS include this disclaimer at the end: "
        "\"This is educational only and not a substitute for medical advice. "
        "Consult a urologist for diagnosis and treatment.\""
    )

def build_user_prompt(context_docs: List[str], question: str) -> str:
    context_text = "\n\n".join(
        [f"[Doc {i+1}]\n{doc}" for i, doc in enumerate(context_docs)]
    )
    return (
        f"CONTEXT (urology guideline excerpts):\n{context_text}\n\n"
        f"PATIENT QUESTION:\n{question}\n\n"
        "INSTRUCTIONS:\n"
        "- Cite which [Doc #] you used inline where relevant.\n"
        "- If multiple options exist, outline them simply (what it is, who it's for, common risks).\n"
        "- If urgent red flags are mentioned (fever with urinary symptoms, inability to pass urine, severe pain, visible blood clots), "
        "advise urgent medical attention.\n"
    )

def pretty_print_hits(hits):
    print("\n📚 Retrieved passages:")
    for i, h in enumerate(hits, 1):
        preview = h.page_content[:220].replace("\n", " ")
        print(f"  [{i}] {preview}...")

def main():
    # Optional: work around macOS OpenMP duplicate runtime crashes with FAISS
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Require API key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='sk-proj-...'")

    # Load retriever and OpenAI client
    db = load_retriever()
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    client = OpenAI()

    print("✅ Urology RAG chatbot ready. Type your question (or 'exit').")
    while True:
        try:
            q = input("\n❓ Your question: ").strip()
            if not q or q.lower() in {"exit", "quit"}:
                print("👋 Bye!")
                break

            # Retrieve top-k chunks
            docs = retriever.get_relevant_documents(q)
            pretty_print_hits(docs)

            context_docs = [d.page_content for d in docs]
            system_msg = make_system_prompt()
            user_msg = build_user_prompt(context_docs, q)

            # Generate answer grounded in context
            resp = client.chat.completions.create(
                model=GEN_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            answer = resp.choices[0].message.content
            print("\n🩺 Answer:\n")
            print(textwrap.fill(answer, width=100))
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}")

if __name__ == "__main__":
    main()

