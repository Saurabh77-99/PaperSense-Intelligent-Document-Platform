# src/data_processing/classify.py
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_DIR = os.getenv("PROJECT_ROOT", "/content/drive/MyDrive/PaperSense-Intelligent-Document-Platform")
MODEL_DIR = os.path.join(MODEL_DIR, "models")

def build_baseline_pipeline():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    return pipe

def train_baseline(texts, labels, save_path=None):
    pipe = build_baseline_pipeline()
    pipe.fit(texts, labels)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pipe, save_path)
    return pipe

def load_model(path):
    return joblib.load(path)

def predict_texts(pipe, texts):
    probs = pipe.predict_proba(texts)
    preds = pipe.predict(texts)
    return preds, probs

if __name__ == "__main__":
    print("Classifier module ready — use train_baseline(texts, labels, save_path).")
