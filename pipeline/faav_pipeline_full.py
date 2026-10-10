"""
FAAV-Full: Faithfulness-Aware Arabic Verification Pipeline
3-class output: Real / Fake / Unverified

Pipeline:
  1. BM25 Retriever       — retrieves top-K passages from training corpus
  2. MARBERT NLI verifier — fine-tuned on four_dataset_training.csv
                            predicts Real / Fake / Unverified per (claim, evidence) pair
  3. Faithfulness layer   — multilingual MiniLM (SentenceTransformers)
       a. Semantic similarity    : cosine(claim_emb, evidence_emb)
       b. Contradiction detection: NLI confidence score for Fake class
       c. Evidence grounding     : cosine(explanation_emb, evidence_emb)
  4. Final decision gate  - accepts Real/Fake only when evidence quality,
                            verifier confidence, similarity, and grounding pass;
                            otherwise abstains to Unverified

Retriever  : BM25Okapi (rank_bm25)
Generator  : MARBERT fine-tuned NLI (UBC-NLP/MARBERTv2)
Faithfulness: paraphrase-multilingual-MiniLM-L12-v2
"""
import os, re, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_FILE, TEST_FILE, TEST_SHEET, RESULTS_DIR,
    FAAV_FULL_CSV, FAAV_FULL_METRICS,
    MARBERT_SAVE,
)
from utils.faav_common import faithfulness_controlled_decision

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
)

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("WARNING: rank_bm25 not installed — skipping BM25 retrieval. pip install rank-bm25")

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("WARNING: sentence-transformers not installed. pip install sentence-transformers")

# ── Config ────────────────────────────────────────────────────────────────────
NLI_BASE       = "UBC-NLP/MARBERTv2"
SENT_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NLI_MODEL_DIR  = os.path.join(RESULTS_DIR, "faav_nli_model")
MAX_LEN        = 128
BATCH_SIZE     = 32
EPOCHS         = 3
LR             = 2e-5
CORPUS_LIMIT   = 50_000
TOP_K          = 5
FAITH_THRESH   = float(os.environ.get("FAITH_THRESH", "0.30"))
EVIDENCE_SIM_THRESH = float(os.environ.get("EVIDENCE_SIM_THRESH", "0.20"))
NLI_CONF_THRESH = float(os.environ.get("NLI_CONF_THRESH", "0.50"))
MIN_EVIDENCE_CHARS = int(os.environ.get("MIN_EVIDENCE_CHARS", "40"))
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS         = ["Real", "Fake", "Unverified"]


def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_bm25(text):
    return normalize_arabic(text).split()


# ── Dataset ───────────────────────────────────────────────────────────────────
class NLIDataset(Dataset):
    def __init__(self, claims, evidences, labels, tokenizer, max_len):
        self.claims    = claims
        self.evidences = evidences
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.claims)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.claims[idx], self.evidences[idx],
            max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Train NLI model ───────────────────────────────────────────────────────────
def train_nli_model(save_dir):
    print("Loading training data for NLI fine-tuning ...")
    df = pd.read_csv(TRAIN_FILE, usecols=["text", "label", "evidence_text"],
                     low_memory=False)
    df = df[df["label"].isin(LABELS)].dropna(subset=["text", "label"])
    df["evidence_text"] = df["evidence_text"].fillna("")
    df["claim_norm"]    = df["text"].apply(normalize_arabic)
    df["ev_norm"]       = df["evidence_text"].apply(normalize_arabic)

    le = LabelEncoder().fit(LABELS)
    df["label_id"] = le.transform(df["label"])
    num_classes = len(le.classes_)

    print(f"NLI training: {len(df)} rows | {df['label'].value_counts().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(NLI_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_BASE, num_labels=num_classes, ignore_mismatched_sizes=True
    ).to(DEVICE)

    train_dl = DataLoader(
        NLIDataset(df["claim_norm"].tolist(), df["ev_norm"].tolist(),
                   df["label_id"].tolist(), tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )

    counts  = Counter(df["label_id"].tolist())
    total   = sum(counts.values())
    class_w = torch.tensor(
        [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float, device=DEVICE,
    )
    loss_fn   = nn.CrossEntropyLoss(weight=class_w)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    print(f"Fine-tuning MARBERT NLI on {DEVICE} for {EPOCHS} epochs ...")
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for batch in train_dl:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labs = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={running/len(train_dl):.4f}")

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "label_classes.json"), "w") as f:
        json.dump(le.classes_.tolist(), f)
    print(f"NLI model saved -> {save_dir}")
    return model, tokenizer, le


# ── Load or train NLI model ───────────────────────────────────────────────────
label_classes_path = os.path.join(NLI_MODEL_DIR, "label_classes.json")
if os.path.exists(label_classes_path):
    print(f"Loading saved NLI model from {NLI_MODEL_DIR} ...")
    tokenizer  = AutoTokenizer.from_pretrained(NLI_MODEL_DIR)
    nli_model  = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_DIR).to(DEVICE)
    with open(label_classes_path) as f:
        _classes = json.load(f)
    le = LabelEncoder()
    le.fit(_classes)
else:
    nli_model, tokenizer, le = train_nli_model(NLI_MODEL_DIR)

nli_model.eval()
num_classes = len(le.classes_)
fake_idx = le.transform(["Fake"])[0]

# ── BM25 index ────────────────────────────────────────────────────────────────
bm25 = None
corpus_texts = []
if HAS_BM25:
    print(f"\nBuilding BM25 index (corpus cap={CORPUS_LIMIT:,}) ...")
    df_corpus    = pd.read_csv(TRAIN_FILE, usecols=["text"], low_memory=False).dropna()
    corpus_texts = df_corpus["text"].head(CORPUS_LIMIT).tolist()
    bm25         = BM25Okapi([tokenize_bm25(t) for t in corpus_texts])
    print(f"BM25 ready — {len(corpus_texts):,} documents")

# ── SentenceTransformer for faithfulness ─────────────────────────────────────
sent_model = None
if HAS_SBERT:
    print(f"\nLoading SentenceTransformer: {SENT_MODEL} ...")
    sent_model = SentenceTransformer(SENT_MODEL)

# ── Load test set ─────────────────────────────────────────────────────────────
print(f"\nLoading test set ...")
df_test = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
print(f"Running FAAV-Full on {len(df_test)} claims ...")

claims_norm = [normalize_arabic(str(c)) for c in df_test["claim_ar"]]
evs_norm    = [normalize_arabic(str(e)) for e in df_test["evidence_text"].fillna("")]

# ── BM25 retrieval: augment evidence per claim ────────────────────────────────
augmented_evidences = []
bm25_scores_list    = []
for i, (claim_norm, existing_ev) in enumerate(zip(claims_norm, evs_norm)):
    if bm25 is not None:
        q_tokens  = tokenize_bm25(claim_norm)
        scores    = bm25.get_scores(q_tokens)
        top_idx   = np.argsort(scores)[::-1][:TOP_K]
        top_score = float(scores[top_idx[0]]) if len(top_idx) > 0 else 0.0
        retrieved  = " ".join(corpus_texts[j] for j in top_idx)
        # Blend: BM25 retrieved + existing evidence
        combined  = (existing_ev + " " + retrieved).strip() if existing_ev else retrieved
        bm25_scores_list.append(round(top_score, 4))
    else:
        combined  = existing_ev
        bm25_scores_list.append(0.0)
    augmented_evidences.append(combined)

# ── NLI batch inference ───────────────────────────────────────────────────────
print("Running NLI inference (batched) ...")
nli_preds, contradiction_scores, all_proba = [], [], []
with torch.no_grad():
    for i in range(0, len(claims_norm), BATCH_SIZE):
        bc = claims_norm[i:i+BATCH_SIZE]
        be = augmented_evidences[i:i+BATCH_SIZE]
        enc = tokenizer(
            bc, be, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        logits = nli_model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).logits
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(proba, axis=1)
        nli_preds.extend(le.inverse_transform(preds))
        contradiction_scores.extend(proba[:, fake_idx].tolist())
        all_proba.extend(proba.tolist())

print(f"NLI done. dist={Counter(nli_preds)}")

# ── Faithfulness scoring per claim ───────────────────────────────────────────
results = []
for idx, row in df_test.iterrows():
    claim_norm = claims_norm[idx]
    ev_norm    = augmented_evidences[idx]
    nli_pred   = nli_preds[idx]
    contra_sc  = contradiction_scores[idx]
    nli_conf   = float(max(all_proba[idx]))

    # Generate the verifier explanation before the final evidence-quality gate.
    if nli_pred == "Fake":
        verifier_explanation = "The evidence contradicts or refutes the claim."
    elif nli_pred == "Unverified":
        verifier_explanation = "The evidence does not clearly confirm or deny the claim."
    else:
        verifier_explanation = "The evidence supports or is consistent with the claim."

    # Faithfulness metrics via SentenceTransformers
    if sent_model is not None:
        c_emb    = sent_model.encode(claim_norm,  convert_to_tensor=True)
        ev_emb   = sent_model.encode(ev_norm[:512], convert_to_tensor=True)
        ex_emb   = sent_model.encode(verifier_explanation, convert_to_tensor=True)

        # a. Semantic similarity: claim vs evidence
        sem_sim  = float(st_util.cos_sim(c_emb, ev_emb).item())

        # b. Contradiction detection score: NLI Fake class probability
        contra   = round(contra_sc, 4)

        # c. Evidence grounding: explanation vs evidence
        grounding = float(st_util.cos_sim(ex_emb, ev_emb).item())

        faith_label = "High" if grounding >= FAITH_THRESH else "Low"
    else:
        sem_sim   = 0.0
        contra    = round(contra_sc, 4)
        grounding = 0.0
        faith_label = "Low"

    prediction, decision_reason = faithfulness_controlled_decision(
        nli_pred,
        ev_norm,
        verifier_confidence=nli_conf,
        semantic_similarity=sem_sim,
        grounding_score=grounding,
        min_evidence_chars=MIN_EVIDENCE_CHARS,
        min_verifier_confidence=NLI_CONF_THRESH,
        min_semantic_similarity=EVIDENCE_SIM_THRESH,
        min_grounding_score=FAITH_THRESH,
    )
    decision_overridden = prediction != nli_pred
    if decision_overridden:
        explanation = (
            "The system abstained because the available evidence or the "
            "faithfulness checks did not meet the required thresholds."
        )
    else:
        explanation = verifier_explanation

    results.append({
        "claim_ar":               row["claim_ar"],
        "gold_label":             row["gold_label"],
        "faav_prediction":        prediction,
        "verifier_prediction":    nli_pred,
        "nli_label":              nli_pred,
        "verifier_confidence":    round(nli_conf, 4),
        "verifier_explanation":   verifier_explanation,
        "explanation":            explanation,
        "bm25_top_score":         bm25_scores_list[idx],
        "semantic_similarity":    round(sem_sim,   4),
        "contradiction_score":    round(contra,    4),
        "grounding_score":        round(grounding, 4),
        "predicted_faithfulness": faith_label,
        "final_decision_reason":  decision_reason,
        "decision_overridden":    decision_overridden,
        "is_correct":             prediction == row["gold_label"],
    })

df_out = pd.DataFrame(results)

# ── Classification metrics ─────────────────────────────────────────────────
y_true = df_out["gold_label"].tolist()
y_pred = df_out["faav_prediction"].tolist()

acc    = accuracy_score(y_true, y_pred)
prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec    = recall_score(y_true, y_pred, average="macro", zero_division=0)
mf1    = f1_score(y_true, y_pred, average="macro", zero_division=0)
wf1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
mcc    = matthews_corrcoef(y_true, y_pred)
kappa  = cohen_kappa_score(y_true, y_pred)
report = classification_report(y_true, y_pred, zero_division=0)
matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
try:
    auc = roc_auc_score(
        pd.get_dummies(y_true).values,
        np.array(all_proba), multi_class="ovr", average="macro",
    )
except Exception:
    auc = float("nan")

# ── Verification metrics ──────────────────────────────────────────────────
faith_acc    = accuracy_score(df_test["faithfulness_label"], df_out["predicted_faithfulness"])
halluc_rate  = (df_out["predicted_faithfulness"] == "Low").mean()
avg_sem_sim  = df_out["semantic_similarity"].mean()
avg_grounding= df_out["grounding_score"].mean()
avg_contra   = df_out["contradiction_score"].mean()
abstention_rate = (df_out["faav_prediction"] == "Unverified").mean()
override_rate = df_out["decision_overridden"].mean()

if "retrieval_quality" in df_test.columns:
    rq = df_test["retrieval_quality"].value_counts(normalize=True) * 100
    ev_relevance = (rq.get("Relevant", 0) * 1.0 +
                    rq.get("Partially Relevant", 0) * 0.5) / 100.0
else:
    ev_relevance = float("nan")

# ── Print summary ─────────────────────────────────────────────────────────
print(f"\n=== FAAV-Full Results ===")
print(f"Accuracy              : {acc:.4f}")
print(f"Macro Precision       : {prec:.4f}")
print(f"Macro Recall          : {rec:.4f}")
print(f"Macro F1              : {mf1:.4f}")
print(f"Weighted F1           : {wf1:.4f}")
print(f"MCC                   : {mcc:.4f}")
print(f"Cohen Kappa           : {kappa:.4f}")
print(f"AUC-ROC               : {auc:.4f}")
print(f"\n--- Verification Metrics ---")
print(f"Evidence Relevance    : {ev_relevance:.4f}")
print(f"Faithfulness Accuracy : {faith_acc:.4f}")
print(f"Hallucination Rate    : {halluc_rate:.4f}")
print(f"Avg Semantic Sim      : {avg_sem_sim:.4f}")
print(f"Avg Contradiction Sc. : {avg_contra:.4f}")
print(f"Avg Grounding Score   : {avg_grounding:.4f}")
print(f"Final Abstention Rate : {abstention_rate:.4f}")
print(f"Verifier Override Rate: {override_rate:.4f}")
print(f"\n{report}")
print(f"Confusion Matrix:\n{matrix}")

# ── Save ──────────────────────────────────────────────────────────────────────
df_out.to_csv(FAAV_FULL_CSV, index=False, encoding="utf-8-sig")

with open(FAAV_FULL_METRICS, "w", encoding="utf-8") as f:
    f.write("FAAV-Full Pipeline Results\n")
    f.write("==========================\n\n")
    f.write(f"Retriever   : BM25Okapi  (corpus={len(corpus_texts):,}, top-k={TOP_K})\n")
    f.write(f"NLI model   : {NLI_BASE} fine-tuned (3-class: Real/Fake/Unverified)\n")
    f.write(f"Faithfulness: {SENT_MODEL}\n")
    f.write(f"Device      : {DEVICE}\n\n")
    f.write("Final decision gate:\n")
    f.write(f"  Minimum evidence characters : {MIN_EVIDENCE_CHARS}\n")
    f.write(f"  Minimum verifier confidence : {NLI_CONF_THRESH:.2f}\n")
    f.write(f"  Minimum semantic similarity : {EVIDENCE_SIM_THRESH:.2f}\n")
    f.write(f"  Minimum grounding score     : {FAITH_THRESH:.2f}\n\n")
    f.write("=== CLASSIFICATION METRICS ===\n")
    f.write(f"Accuracy        : {acc:.4f}\n")
    f.write(f"Macro Precision : {prec:.4f}\n")
    f.write(f"Macro Recall    : {rec:.4f}\n")
    f.write(f"Macro F1        : {mf1:.4f}\n")
    f.write(f"Weighted F1     : {wf1:.4f}\n")
    f.write(f"MCC             : {mcc:.4f}\n")
    f.write(f"Cohen Kappa     : {kappa:.4f}\n")
    f.write(f"AUC-ROC (macro) : {auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"\nConfusion Matrix (Real/Fake/Unverified):\n{matrix}\n\n")
    f.write("=== VERIFICATION METRICS ===\n")
    f.write(f"Evidence Relevance Score  : {ev_relevance:.4f}\n")
    f.write(f"Faithfulness Accuracy     : {faith_acc:.4f}\n")
    f.write(f"Hallucination Rate        : {halluc_rate:.4f}\n")
    f.write(f"Avg Semantic Similarity   : {avg_sem_sim:.4f}\n")
    f.write(f"Avg Contradiction Score   : {avg_contra:.4f}\n")
    f.write(f"Avg Evidence Grounding    : {avg_grounding:.4f}\n")
    f.write(f"Final Abstention Rate     : {abstention_rate:.4f}\n")
    f.write(f"Verifier Override Rate    : {override_rate:.4f}\n")

print(f"\nResults saved -> {FAAV_FULL_CSV}")
print(f"Metrics saved -> {FAAV_FULL_METRICS}")
