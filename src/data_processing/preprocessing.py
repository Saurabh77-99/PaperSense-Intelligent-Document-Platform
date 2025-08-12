# src/data_processing/preprocessing.py
import os
import re
from PIL import Image
import pytesseract
import nltk
from nltk.corpus import stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def ocr_image_to_text(image_path, lang='eng'):
    """
    Extract text from an image using pytesseract.
    """
    img = Image.open(image_path)
    raw = pytesseract.image_to_string(img, lang=lang)
    return raw

def clean_text(text, lower=True, remove_stopwords=True):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    if lower:
        text = text.lower()
    text = re.sub(r'[^0-9a-zA-Z\s\.\,]', '', text)
    if remove_stopwords:
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in STOPWORDS]
        return " ".join(tokens)
    return text

def extract_text_from_pdf_pages(image_paths, lang='eng'):
    texts = []
    for p in image_paths:
        t = ocr_image_to_text(p, lang=lang)
        texts.append(clean_text(t))
    return "\n".join(texts)

if __name__ == "__main__":
    print("Preprocessing module ready.")
