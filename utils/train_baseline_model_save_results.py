import os
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from config import DATA_DIR, RESULTS_DIR, TRAIN_FILE, TEST_FILE, TEST_SHEET
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


RESULTS_FILE = os.path.join(RESULTS_DIR, "baseline_model_results.txt")
PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "baseline_model_predictions.csv")

df = pd.read_csv(INPUT_FILE)

df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df[df["content"].str.strip() != ""]

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("Baseline Model Results\n")
    f.write("======================\n\n")
    f.write(f"Total dataset rows: {len(df)}\n")
    f.write(f"Training rows: {len(X_train)}\n")
    f.write(f"Testing rows: {len(X_test)}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))

predictions_df = pd.DataFrame({
    "text": X_test,
    "actual_label": y_test,
    "predicted_label": y_pred
})

predictions_df.to_csv(PREDICTIONS_FILE, index=False, encoding="utf-8-sig")

print("Done.")
print(f"Results saved to: {RESULTS_FILE}")
print(f"Predictions saved to: {PREDICTIONS_FILE}")