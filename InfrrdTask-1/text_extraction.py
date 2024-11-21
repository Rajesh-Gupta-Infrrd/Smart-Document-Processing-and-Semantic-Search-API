import pdfplumber
import pytesseract
from PIL import Image

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith((".jpg", ".jpeg", ".png")):
        return extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file format")

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_image(file_path):
    with Image.open(file_path) as img:
        return pytesseract.image_to_string(img)