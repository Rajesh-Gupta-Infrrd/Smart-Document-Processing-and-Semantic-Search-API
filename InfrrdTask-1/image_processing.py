import cv2
import numpy as np

def detect_lines(image_path):
    """Detects all lines present in the image using Hough Line Transform."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    line_metadata = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            line_metadata.append({"start": (x1, y1), "end": (x2, y2)})

    return line_metadata


def detect_words(image_path):
    """Detects all textual words present in the image without OCR using contours."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    words_metadata = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Minimum size filter
            words_metadata.append({"bounding_box": (x, y, w, h)})

    return words_metadata


def process_image(image_path):
    """Processes the image and returns line and word metadata."""
    lines = detect_lines(image_path)
    words = detect_words(image_path)
    return {"lines": lines, "words": words}

