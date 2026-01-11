import pandas as pd
from rapidfuzz import process, fuzz
import os

# Absolute path to dataset (SAFE for Colab + local)
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "../data/medical_text_real/A_Z_medicines_dataset_of_India.csv"
)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Normalize medicine names
medicine_names = (
    df["name"]
    .astype(str)
    .str.lower()
    .dropna()
    .unique()
    .tolist()
)

def find_best_match(ocr_text, threshold=70):
    """
    Finds best matching medicine name from OCR text
    by checking each line separately.
    """
    if not ocr_text:
        return None, 0

    ocr_text = ocr_text.lower()
    lines = ocr_text.split("\n")

    best_match = None
    best_score = 0

    for line in lines:
        line = line.strip()
        if len(line) < 4:
            continue

        match, score, _ = process.extractOne(
            line,
            medicine_names,
            scorer=fuzz.partial_ratio
        )

        if score > best_score:
            best_match = match
            best_score = score

    if best_score >= threshold:
        return best_match, best_score

    return None, best_score



# 🔍 Standalone test
if __name__ == "__main__":
    test_texts = [
        "Paracelamol 650 mg",
        "Crocn tablet",
        "Azithromicin"
    ]

    for text in test_texts:
        result, score = find_best_match(text)
        print(f"OCR: {text} → Match: {result}, Score: {score}")
