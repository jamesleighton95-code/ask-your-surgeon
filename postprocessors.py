import re

def enforce_exact_risks(answer: str, retrieved_chunks: list[str], debug: bool = False) -> str:
    """
    Ensures risk statements in the answer exactly match wording from the retrieved context.
    Replaces paraphrased risks with the original PDF wording if found.
    If debug=True, prints a warning when corrections are made.
    """

    context_text = " ".join(retrieved_chunks)
    risk_pattern = re.compile(r"- (.+?) → Risk: (.+)")

    corrected_lines = []
    corrections_made = False

    for line in answer.splitlines():
        match = risk_pattern.match(line.strip())
        if match:
            complication, model_risk = match.groups()

            if complication in context_text:
                comp_idx = context_text.find(complication)
                snippet = context_text[comp_idx: comp_idx + 500]



                risk_match = re.search(
                    r"(?:\d+ in \d+"
                    r"|less than \d+ in \d+"
                    r"|fewer than \d+ in \d+"
                    r"|up to \d+ in \d+"                   
                    r"|between \d+ in \d+ and \d+ in \d+"
                    r"|between \d+ and \d+ in \d+"
                    r"|\d+%|\d+-\d+%"                    
                    r"|almost all"
                    r"|most"
                    r"|majority)",
                    snippet,
                    re.IGNORECASE
                )


                if risk_match:
                    corrected_risk = risk_match.group(0)
                    corrected_line = f"- {complication} → Risk: {corrected_risk}"

                    if corrected_line.strip() != line.strip():
                        corrections_made = True
                        if debug:
                            print(f"[DEBUG] Corrected risk for '{complication}':")
                            print(f"    LLM said → {model_risk}")
                            print(f"    PDF says → {corrected_risk}")

                    corrected_lines.append(corrected_line)
                else:
                    corrected_lines.append(line)
            else:
                corrected_lines.append(line)
        else:
            corrected_lines.append(line)

    if corrections_made and debug:
        print("[DEBUG] One or more risk statements were corrected to match PDF wording.")

    return "\n".join(corrected_lines)

