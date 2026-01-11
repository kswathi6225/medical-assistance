import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import pytesseract

from utils.image_quality_check import is_image_clear
from models.name_matching import find_best_match


IMAGE_DIR = os.path.join(
    os.path.dirname(__file__),
    "../data/medicine_images"
)

for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not is_image_clear(img_path):
        print(f"{img_name}: Image too blurry")
        continue

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    ocr_text = pytesseract.image_to_string(gray)
    medicine, score = find_best_match(ocr_text)

    print(f"\n--- {img_name} ---")
    print("OCR:", ocr_text)
    if medicine:
        print(f"Matched medicine: {medicine} ({score}%)")
    else:
        print("No confident match")
