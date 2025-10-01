import sys
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def inspect_index(index_path, query, k=5, show_full=False):
    """Load a FAISS index and inspect the most relevant docs for a query."""
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": k})

    docs = retriever.get_relevant_documents(query)
    print(f"\n🔎 Retrieved documents for query: '{query}'\n")
    for i, d in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        if show_full:
            print(d.page_content)  # dump entire chunk
        else:
            print(d.page_content[:800])  # truncated for readability
        print(f"[Source: {d.metadata.get('source', '?')}]\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python inspect_index.py <condition> <query> [--full]")
        print("Example: python inspect_index.py bph 'risks of HoLEP' --full")
    else:
        condition = sys.argv[1]
        query_parts = []
        show_full = False
        for arg in sys.argv[2:]:
            if arg == "--full":
                show_full = True
            else:
                query_parts.append(arg)
        query = " ".join(query_parts)
        index_path = f"embeddings/{condition}_index"
        inspect_index(index_path, query, show_full=show_full)

