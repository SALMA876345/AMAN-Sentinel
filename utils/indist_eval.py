"""
PROPER in-distribution evaluation of the framework: hold out 20% of the claim
training data (same distribution), train on 80%, test on 20%. Deduplicated,
stratified, no leakage. Shows what the models really do on matched data.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

LABELS = ["Real", "Fake"]
df = pd.read_csv("data/claim_training_2class.csv",
                 usecols=["text", "label", "evidence_text"], low_memory=False).dropna(subset=["text", "label"])
df["evidence_text"] = df["evidence_text"].fillna("")
df = df[df["label"].isin(LABELS)]
df["t"] = df["text"].astype(str).str.strip()
df = df.drop_duplicates(subset=["t"])                      # no duplicate claims across split
print(f"Claims (deduped): {len(df)}  {df['label'].value_counts().to_dict()}")

tr, te = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=42)
print(f"train {len(tr)}  test {len(te)}  (disjoint, stratified)\n")

def run(name, clf, use_ev):
    Xtr = tr["t"] + (" " + tr["evidence_text"] if use_ev else "")
    Xte = te["t"] + (" " + te["evidence_text"] if use_ev else "")
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
                     ("clf", clf)])
    pipe.fit(Xtr, tr["label"])
    yp = pipe.predict(Xte)
    print(f"{name:<34} Acc={accuracy_score(te['label'],yp):.4f}  "
          f"MacroF1={f1_score(te['label'],yp,average='macro',labels=LABELS):.4f}  "
          f"MCC={matthews_corrcoef(te['label'],yp):.4f}")

print("=== IN-DISTRIBUTION (train 80% -> test 20% of the framework's own claims) ===")
run("LR  (claim only)",       LogisticRegression(max_iter=2000, class_weight="balanced"), False)
run("LR  (claim + evidence)", LogisticRegression(max_iter=2000, class_weight="balanced"), True)
run("SVM (claim only)",       LinearSVC(class_weight="balanced"), False)
run("SVM (claim + evidence)", LinearSVC(class_weight="balanced"), True)
print("\n(MARBERT on this split would run on Kaggle and should land higher still.)")
