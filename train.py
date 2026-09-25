import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Project folder
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Correct dataset path
DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "emotion_dataset.csv"
)

# Model folder
MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "emotion_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)


print("Project folder:")
print(BASE_DIR)

print("\nDataset path:")
print(DATASET_PATH)

print("\nDataset exists:")
print(os.path.exists(DATASET_PATH))


# Load dataset
data = pd.read_csv(
    DATASET_PATH,
    encoding="utf-8"
)

print("\nDataset loaded successfully!")
print("Number of records:", len(data))

print("\nColumns:")
print(data.columns.tolist())