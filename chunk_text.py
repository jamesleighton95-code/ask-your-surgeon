from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Input: your cleaned guideline
input_file = "data/clean/EAU_Prostate_Cancer_cleaned.txt"
output_file = "data/clean/EAU_Prostate_Cancer_chunks.txt"

# Load text
with open(input_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Token counter (so chunks are ~1000 tokens each)
encoding = tiktoken.encoding_for_model("gpt-4o-mini")  # works for GPT models
def tok_len(text): return len(encoding.encode(text))

# Splitter: 1000-token chunks with 200-token overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=tok_len
)

chunks = splitter.split_text(raw_text)

# Save chunks into a file
with open(output_file, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks, 1):
        f.write(f"--- Chunk {i} ---\n")
        f.write(chunk + "\n\n")

print(f"✅ Done! Created {len(chunks)} chunks and saved to {output_file}")

