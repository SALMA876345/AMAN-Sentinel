"""
Fast local proxy: does adding evidence_text help cross-domain?
Train LR (TF-IDF) on claim-only vs claim+evidence, test on UAE-MVD.
TF-IDF is a LOWER BOUND — MARBERT models semantics, so it can do better than this.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

LABELS = ["Real", "Fake"]

tr = pd.read_csv("data/claim_training_2class.csv",
                 usecols=["text", "label", "evidence_text"], low_memory=False).dropna(subset=["text", "label"])
tr["evidence_text"] = tr["evidence_text"].fillna("")
te = pd.read_excel("data/uae_test_2class.xlsx", sheet_name="Tabelle1")
te = te[te["gold_label"].isin(LABELS)].copy()
te["claim_ar"] = [str(x) for x in te["claim_ar"].tolist()]
te["evidence_text"] = [str(x) if x is not None else "" for x in te["evidence_text"].tolist()]

def run(name, Xtr, ytr, Xte, yte):
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    print(f"{name:<28} Acc={accuracy_score(yte,yp):.4f}  "
          f"MacroF1={f1_score(yte,yp,average='macro',labels=LABELS):.4f}  "
          f"MCC={matthews_corrcoef(yte,yp):.4f}")

print("=== Cross-domain: does evidence help? (TF-IDF LR, lower bound) ===")
run("claim only",
    tr["text"], tr["label"], te["claim_ar"], te["gold_label"])
run("claim + evidence",
    tr["text"] + " " + tr["evidence_text"], tr["label"],
    te["claim_ar"] + " " + te["evidence_text"], te["gold_label"])
run("evidence only",
    tr["evidence_text"], tr["label"], te["evidence_text"], te["gold_label"])
