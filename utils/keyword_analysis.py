"""
Show the KEYWORDS each TF-IDF+LR model learns for Real vs Fake, in:
  (A) the cross-domain TRAINING corpus (claim_training_2class.csv)
  (B) the in-domain UAE-MVD corpus (claim_ar)
and measure how much the training signal words even appear in the test corpus.
Output (Arabic) -> results/keyword_analysis.txt
"""
import io
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

out = io.StringIO()
def P(*a): print(*a, file=out)

def top_words(texts, labels, n=25):
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 1), min_df=3)
    X = vec.fit_transform(texts)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, labels)
    vocab = vec.get_feature_names_out()
    coef = lr.coef_[0]                 # classes_ alphabetical: ['Fake','Real']
    order = coef.argsort()             # most negative -> Fake ; most positive -> Real
    fake = [vocab[i] for i in order[:n]]
    real = [vocab[i] for i in order[::-1][:n]]
    return real, fake, set(vocab)

# (A) cross-domain training
tr = pd.read_csv("data/claim_training_2class.csv", usecols=["text", "label"]).dropna()
A_real, A_fake, A_vocab = top_words(tr["text"].astype(str), tr["label"])

# (B) in-domain UAE-MVD
te = pd.read_excel("data/uae_test_dataset_full30k.xlsx", sheet_name="Tabelle1")
te = te[te["gold_label"].isin(["Real", "Fake"])].dropna(subset=["claim_ar"])
B_real, B_fake, B_vocab = top_words(te["claim_ar"].astype(str), te["gold_label"])

P("=" * 70)
P("(A) CROSS-DOMAIN TRAINING corpus  (claim_training_2class.csv)")
P("    What the model learns as the signal words:")
P("-" * 70)
P("TOP 'REAL' keywords:\n   " + "  ".join(A_real))
P("\nTOP 'FAKE' keywords:\n   " + "  ".join(A_fake))

P("\n" + "=" * 70)
P("(B) IN-DOMAIN UAE-MVD corpus  (claim_ar)")
P("    Completely different signal words:")
P("-" * 70)
P("TOP 'REAL' keywords:\n   " + "  ".join(B_real))
P("\nTOP 'FAKE' keywords:\n   " + "  ".join(B_fake))

# overlap: do the training signal words even appear in the test vocabulary?
A_signal = set(A_real) | set(A_fake)
present = A_signal & B_vocab
P("\n" + "=" * 70)
P("WHY CROSS-DOMAIN FAILS — vocabulary overlap")
P("-" * 70)
P(f"Training vocab size: {len(A_vocab):,}   Test vocab size: {len(B_vocab):,}")
shared = A_vocab & B_vocab
P(f"Shared vocabulary:   {len(shared):,} words "
  f"({100*len(shared)/len(A_vocab):.1f}% of training vocab)")
P(f"\nOf the {len(A_signal)} TOP training signal words, "
  f"{len(present)} appear in the test corpus at all:")
P("   present: " + ("  ".join(sorted(present)) if present else "(none)"))
P("   MISSING: " + "  ".join(sorted(A_signal - present)))
P("\n=> The words the model relies on to detect Fake/Real in training are")
P("   largely ABSENT from the UAE test claims, so it cannot transfer.")

with open("results/keyword_analysis.txt", "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("Saved -> results/keyword_analysis.txt")
