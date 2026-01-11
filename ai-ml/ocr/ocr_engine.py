import cv2
import pytesseract

def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("OCR: Image load failed")
        return ""

    # Resize for better OCR
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold works better for medicine strips
    gray = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    # OCR configuration (IMPORTANT)
    custom_config = r'--oem 3 --psm 6'

    text = pytesseract.image_to_string(gray, config=custom_config)

    return text.strip()
