import pandas as pd
from rapidfuzz import process, fuzz

# Load medicine names from REAL dataset
DATA_PATH = "../data/medical_text_real/medicines_master.csv"

df = pd.read_csv(DATA_PATH)

# Normalize medicine names
medicine_names = df["name"].str.lower().dropna().unique().tolist()

def find_best_match(ocr_text, threshold=80):
    """
    Finds best matching medicine name from dataset
    """
    ocr_text = ocr_text.lower()

    match, score, _ = process.extractOne(
        ocr_text,
        medicine_names,
        scorer=fuzz.token_sort_ratio
    )

    if score >= threshold:
        return match, score
    else:
        return None, score


# 🔍 Test example
if __name__ == "__main__":
    test_texts = [
        "Paracelamol 650 mg",
        "Crocn tablet",
        "Azithromicin"
    ]

    for text in test_texts:
        result, score = find_best_match(text)
        print(f"OCR: {text} → Match: {result}, Score: {score}")
