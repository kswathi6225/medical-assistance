import sys
import os

sys.path.append(os.path.dirname(__file__))

from ocr.ocr_engine import extract_text_from_image
from models.name_matching import find_best_match
from services.medicine_info import get_medicine_info

def analyze_medicine_image(image_path):
    ocr_text = extract_text_from_image(image_path)

    medicine, confidence = find_best_match(ocr_text)

    medicine_info = get_medicine_info(medicine)

    return {
        "ocr_text": ocr_text,
        "medicine_name": medicine,
        "confidence": confidence,
        "medicine_info": medicine_info
    }


# 🔍 Test locally / Colab
if __name__ == "__main__":
    test_image = os.path.join(
        os.path.dirname(__file__),
        "data/medicine_images/dolo650_strip.jpg"
    )

    result = analyze_medicine_image(test_image)
    print(result)
