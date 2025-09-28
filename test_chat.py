# test_chat.py
from chatbot import qa_chain

def main():
    print("💬 Ask Your Surgeon Chatbot (type 'quit' to exit)")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        result = qa_chain(query)
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        print("\nAssistant:", answer)
        if sources:
            print("\n📖 Sources used:")
            for i, doc in enumerate(sources, 1):
                print(f"  {i}. {doc.metadata.get('source', 'Unknown source')}")

if __name__ == "__main__":
    main()

