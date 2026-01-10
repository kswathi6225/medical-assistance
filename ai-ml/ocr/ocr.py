import cv2
import pytesseract
import os

IMAGE_DIR = "../data/medicine_images"

for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    
    text = pytesseract.image_to_string(gray)
    print(f"\n--- {img_name} ---")
    print(text)
