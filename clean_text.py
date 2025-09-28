import re

input_file = "data/clean/EAU_Prostate_Cancer.txt"
output_file = "data/clean/EAU_Prostate_Cancer_cleaned.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Remove multiple newlines
text = re.sub(r"\n+", "\n", text)

# Remove page numbers like "Page 12 of 50"
text = re.sub(r"Page \d+ of \d+", "", text)

# Remove multiple spaces
text = re.sub(r"\s+", " ", text)

# Strip leading/trailing whitespace
text = text.strip()

with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print(f"✅ Cleaned text saved to {output_file}")

