import sys, re

if len(sys.argv) != 3:
    print("Usage: python clean_text.py input.txt output.txt")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Remove multiple blank lines
cleaned = re.sub(r"\n\s*\n", "\n\n", text)

# Strip non-ASCII (odd symbols, formatting)
cleaned = re.sub(r"[^\x00-\x7F]+", " ", cleaned)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(cleaned)

print(f"✅ Cleaned {input_file} -> {output_file}")

