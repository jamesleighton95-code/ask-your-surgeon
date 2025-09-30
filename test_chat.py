from chatbot import qa_chain, format_sources

def main():
    print("💬 Ask Your Surgeon Chatbot (type 'quit' to exit)")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        result = qa_chain(query)
        answer = result.get("result", "")
        sources = format_sources(result.get("source_documents", []))

        print("\nAssistant:", answer)
        if sources:
            print("\n📖 Sources used:")
            for i, s in enumerate(sources, 1):
                print(f"  {i}. {s}")

if __name__ == "__main__":
    main()
def format_sources(sources):
    """Convert raw source metadata into human-friendly names."""
    pretty_sources = []
    for doc in sources:
        source = doc.metadata.get("source", "Unknown source")
        print("DEBUG source metadata:", source)   # 👈 add this line
        name = os.path.basename(source)
        pretty_sources.append(SOURCE_MAP.get(name, name))
    return pretty_sources

