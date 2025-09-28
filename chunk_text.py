import sys

if len(sys.argv) != 3:
    print("Usage: python chunk_text.py input.txt output.txt")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

chunk_size = 1000  # characters per chunk
overlap = 200      # overlap between chunks

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

chunks = []
start = 0
while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]
    if chunk.strip():  # only keep non-empty
        chunks.append(chunk)
    start = end - overlap  # step forward with overlap

with open(output_file, "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks, 1):
        f.write(f"--- Chunk {i} ---\n{chunk}\n")

print(f"✅ Chunked {input_file} into {len(chunks)} chunks -> {output_file}")

