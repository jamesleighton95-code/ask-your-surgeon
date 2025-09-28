import sys
import pdfplumber

if len(sys.argv) != 3:
    print("Usage: python convert_one.py input.pdf output.txt")
    sys.exit(1)

input_pdf = sys.argv[1]
output_txt = sys.argv[2]

with pdfplumber.open(input_pdf) as pdf:
    text = "\n".join([page.extract_text() or "" for page in pdf.pages])

with open(output_txt, "w", encoding="utf-8") as f:
    f.write(text)

print(f"✅ Converted {input_pdf} -> {output_txt}")

