import pdfplumber

input_pdf = "data/raw/EAU_Prostate_Cancer.pdf"
output_txt = "data/clean/EAU_Prostate_Cancer.txt"

with pdfplumber.open(input_pdf) as pdf:
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:   # skip blank pages
            text += page_text + "\n"

with open(output_txt, "w", encoding="utf-8") as f:
    f.write(text)

print(f"✅ Done! Saved text to {output_txt}")

