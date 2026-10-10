"""
Assess ANS as a test set:
  1) vocabulary overlap with our training data (predicts transfer)
  2) cross-domain: train on OUR claims -> test on ANS test
  3) in-distribution: train on ANS train -> test on ANS test (Khouja-style)
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

LABELS = ["Real", "Fake"]
def lab(flag): return "Fake" if int(flag) == 1 else "Real"

# our training
tr = pd.read_csv("data/claim_training_2class.csv", usecols=["text", "label"], low_memory=False).dropna()
tr = tr[tr["label"].isin(LABELS)]

# ANS
ans_tr = pd.read_csv("data/ANS/train.csv"); ans_te = pd.read_csv("data/ANS/test.csv")
ans_tr["label"] = ans_tr["fake_flag"].map(lab); ans_te["label"] = ans_te["fake_flag"].map(lab)
print(f"ANS train {len(ans_tr)} {ans_tr['label'].value_counts().to_dict()} | "
      f"test {len(ans_te)} {ans_te['label'].value_counts().to_dict()}")

# 1) vocab overlap
def vocab(texts):
    v = TfidfVectorizer(max_features=30000, min_df=2).fit(texts.astype(str))
    return set(v.get_feature_names_out())
V_tr, V_ans = vocab(tr["text"]), vocab(ans_te["claim_s"])
shared = V_tr & V_ans
print(f"\nVocab overlap: {len(shared)} shared / {len(V_ans)} ANS-test "
      f"({100*len(shared)/len(V_ans):.1f}% of ANS vocab is in our training)")
print("(recall: UAE was ~52% -> failed; >70% predicts good transfer)")

def fit_eval(Xtr, ytr, Xte, yte, name):
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    pipe.fit(Xtr.astype(str), ytr)
    yp = pipe.predict(Xte.astype(str))
    print(f"{name:<40} Acc={accuracy_score(yte,yp):.4f}  "
          f"MacroF1={f1_score(yte,yp,average='macro',labels=LABELS):.4f}  "
          f"MCC={matthews_corrcoef(yte,yp):.4f}")

print("\n=== Transfer tests (LR) ===")
fit_eval(tr["text"], tr["label"], ans_te["claim_s"], ans_te["label"],
         "CROSS-DOMAIN: our train -> ANS test")
fit_eval(ans_tr["claim_s"], ans_tr["label"], ans_te["claim_s"], ans_te["label"],
         "IN-DISTRIBUTION: ANS train -> ANS test")
