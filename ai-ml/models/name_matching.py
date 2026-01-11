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

def find_best_match(ocr_text, threshold=80):
    """
    Finds best matching medicine name from dataset
    """
    if not ocr_text:
        return None, 0

    ocr_text = ocr_text.lower()

    match, score, _ = process.extractOne(
        ocr_text,
        medicine_names,
        scorer=fuzz.token_sort_ratio
    )

    if score >= threshold:
        return match, score
    return None, score


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
