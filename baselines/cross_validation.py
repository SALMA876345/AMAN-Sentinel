"""
5-fold cross-validation for SVM and Logistic Regression baselines.
Reports mean +/- std for accuracy and macro F1 across folds.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_FILE, RESULTS_DIR, CV_METRICS, CV_CSV

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

N_SPLITS   = 5
N_CAP      = 15_000   # per class cap for speed
LABELS     = ["Real", "Fake", "Unverified"]
RANDOM_STATE = 42


def run_cv(name, pipeline, X, y, kf):
    accs, mf1s, mccs, kappas = [], [], [], []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = [X[i] for i in tr_idx], [X[i] for i in val_idx]
        y_tr, y_val = [y[i] for i in tr_idx], [y[i] for i in val_idx]
        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)
        accs.append(accuracy_score(y_val, preds))
        mf1s.append(f1_score(y_val, preds, average="macro", zero_division=0))
        mccs.append(matthews_corrcoef(y_val, preds))
        kappas.append(cohen_kappa_score(y_val, preds))
        print(f"  {name} Fold {fold}: Acc={accs[-1]:.4f}  MacroF1={mf1s[-1]:.4f}")
    return {
        "model": name,
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "macro_f1_mean": np.mean(mf1s), "macro_f1_std": np.std(mf1s),
        "mcc_mean": np.mean(mccs), "mcc_std": np.std(mccs),
        "kappa_mean": np.mean(kappas), "kappa_std": np.std(kappas),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading training data ...")
    df = pd.read_csv(TRAIN_FILE, usecols=["text", "label"], low_memory=False)
    df = df[df["label"].isin(LABELS)].dropna(subset=["text", "label"])

    # Cap per class for speed
    df = pd.concat([
        g.sample(min(len(g), N_CAP), random_state=RANDOM_STATE)
        for _, g in df.groupby("label")
    ]).reset_index(drop=True).sample(frac=1, random_state=RANDOM_STATE)

    print(f"CV training set: {len(df)} rows | {df['label'].value_counts().to_dict()}")

    X = df["text"].astype(str).tolist()
    y = df["label"].tolist()

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    svm_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                  analyzer="char_wb", sublinear_tf=True)),
        ("clf",   LinearSVC(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])

    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                  analyzer="char_wb", sublinear_tf=True)),
        ("clf",   LogisticRegression(class_weight="balanced", max_iter=1000,
                                     solver="lbfgs", random_state=RANDOM_STATE)),
    ])

    print("\n=== SVM 5-fold CV ===")
    svm_results = run_cv("SVM", svm_pipe, X, y, kf)

    print("\n=== LR 5-fold CV ===")
    lr_results = run_cv("LR", lr_pipe, X, y, kf)

    results = [svm_results, lr_results]
    df_cv = pd.DataFrame(results)
    df_cv.to_csv(CV_CSV, index=False)

    with open(CV_METRICS, "w", encoding="utf-8") as f:
        f.write("5-Fold Cross-Validation Results\n")
        f.write("================================\n\n")
        f.write(f"Training rows (capped {N_CAP}/class): {len(df)}\n")
        f.write(f"Label distribution: {df['label'].value_counts().to_dict()}\n\n")
        for r in results:
            f.write(f"--- {r['model']} ---\n")
            f.write(f"Accuracy  : {r['acc_mean']:.4f} +/- {r['acc_std']:.4f}\n")
            f.write(f"Macro F1  : {r['macro_f1_mean']:.4f} +/- {r['macro_f1_std']:.4f}\n")
            f.write(f"MCC       : {r['mcc_mean']:.4f} +/- {r['mcc_std']:.4f}\n")
            f.write(f"Kappa     : {r['kappa_mean']:.4f} +/- {r['kappa_std']:.4f}\n\n")

    print("\n=== CV SUMMARY ===")
    for r in results:
        print(f"{r['model']:<6} Acc={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  "
              f"MacroF1={r['macro_f1_mean']:.4f}+/-{r['macro_f1_std']:.4f}")
    print(f"\nSaved -> {CV_CSV}")


if __name__ == "__main__":
    main()
