# aliases.py

# 🔑 Map keywords in patient queries to the correct BAUS leaflet (PDF filenames)
BAUS_DOC_MAP = {
    "holep": "BAUS_HoLEP.pdf",
    "turp": "BAUS_TURP.pdf",
    "rezum": "BAUS_Rezum.pdf",
    "greenlight": "BAUS_Greenlight.pdf",
    "urolift": "BAUS_Urolift.pdf",
    "aquablation": "BAUS_Aquablation.pdf",
    "bni": "BAUS_BNI.pdf",
    "ralp": "BAUS_RALP.pdf",
    "robotic": "BAUS_RALP.pdf",
    "robotic prostatectomy": "BAUS_RALP.pdf",
    "radical prostatectomy": "BAUS_RALP.pdf",
    "brachytherapy": "BAUS_Brachytherapy.pdf",
    "radiotherapy": "BAUS_Brachytherapy.pdf",
    "prostate biopsy": "BAUS_TP_Biopsies.pdf"


    # Add more as needed...
}

# 🔑 Map keywords in patient queries to the curated ground-truth text files
GROUND_TRUTH_MAP = {
    "holep": "ground_truth/ground_truth_holep.txt",
    "turp": "ground_truth/ground_truth_turp.txt",
    "ralp": "ground_truth/ground_truth_ralp.txt",
    "robotic": "ground_truth/ground_truth_ralp.txt",
    "robotic prostatectomy": "ground_truth/ground_truth_ralp.txt",
    "radical prostatectomy": "ground_truth/ground_truth_ralp.txt",
    "rezum": "ground_truth/ground_truth_REZUM.txt",
    "greenlight": "ground_truth/ground_truth_Greenlight.txt",
    "urolift": "ground_truth/ground_truth_Urolift.txt",
    "aquablation": "ground_truth/ground_truth_Aquablation.txt",
    "bni": "ground_truth/ground_truth_BNI.txt",
    "brachytherapy": "ground_truth/ground_truth_Brachytherapy.txt",
    "radiotherapy": "ground_truth/ground_truth_Brachytherapy.txt",
    "prostate biopsy": "ground_truth/ground_truth_TP_Biopsies.txt"


    # Add more as you prepare ground truth files...
}

