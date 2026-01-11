import cv2

def is_image_clear(image_path, threshold=100):
    """
    Checks if an image is clear using Laplacian variance.
    Returns True if image is clear, False otherwise.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return False

    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var > threshold
