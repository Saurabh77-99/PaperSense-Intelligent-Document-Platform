# src/data_processing/ingestion.py
import os
from pdf2image import convert_from_path
from PIL import Image
import tempfile
import shutil
import pathlib

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/content/drive/MyDrive/PaperSense-Intelligent-Document-Platform")
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def save_uploaded_file(file_like, dest_path):
    """
    Save an in-memory file (werkzeug/FileStorage or Colab upload) to dest_path.
    """
    ensure_dir(os.path.dirname(dest_path))
    with open(dest_path, "wb") as f:
        f.write(file_like.read())
    return dest_path

def pdf_to_images(pdf_path, out_dir=None, dpi=200):
    out_dir = out_dir or (os.path.splitext(pdf_path)[0] + "_pages")
    ensure_dir(out_dir)
    pages = convert_from_path(pdf_path, dpi=dpi)
    paths = []
    for i, page in enumerate(pages):
        p = os.path.join(out_dir, f"page_{i+1}.png")
        page.save(p, "PNG")
        paths.append(p)
    return paths

def image_preprocess_save(image_path, out_path=None, max_dim=1600):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
    out_path = out_path or image_path
    img.save(out_path, format="PNG", optimize=True)
    return out_path

def list_supported_extensions():
    return [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".docx"]

if __name__ == "__main__":
    # quick smoke test (requires a PDF in the datasets folder)
    sample_pdf = os.path.join(DATASETS_DIR, "sample.pdf")
    if os.path.exists(sample_pdf):
        print("Converting sample.pdf → images")
        print(pdf_to_images(sample_pdf))
    else:
        print("Place a sample.pdf in datasets/ to test ingestion.")
