"""
IN-DOMAIN diagnostic: is the low cross-domain score caused by the domain gap?

Train SVM + LR on 80% of UAE-MVD (Real/Fake) and test on the held-out 20% of
UAE-MVD — same distribution for train and test. If scores jump well above the
cross-domain ~0.50, the domain gap is the cause.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

df = pd.read_excel("data/uae_test_dataset_full30k.xlsx", sheet_name="Tabelle1")
df = df[df["gold_label"].isin(["Real", "Fake"])].dropna(subset=["claim_ar"])
print(f"UAE-MVD Real/Fake: {len(df)}  {df['gold_label'].value_counts().to_dict()}")

Xtr, Xte, ytr, yte = train_test_split(
    df["claim_ar"].astype(str), df["gold_label"],
    test_size=0.20, stratify=df["gold_label"], random_state=42)
print(f"train {len(Xtr)}  test {len(Xte)}")

def run(name, clf):
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
                     ("clf", clf)])
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    print(f"{name:<22} Acc={accuracy_score(yte,yp):.4f}  "
          f"MacroF1={f1_score(yte,yp,average='macro'):.4f}  "
          f"MCC={matthews_corrcoef(yte,yp):.4f}")

print("\n=== IN-DOMAIN (train 80% UAE-MVD -> test 20% UAE-MVD) ===")
run("SVM (TF-IDF)", LinearSVC(class_weight="balanced"))
run("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced"))
print("\nCompare: cross-domain SVM=0.5090, LR=0.5264 macroF1")
