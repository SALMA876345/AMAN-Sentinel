# ============================================================
# FAAV - Kaggle Session 2 (single-file version)
# Upload to faav-arabic-fake-news: indist_train.csv + indist_test.xlsx
# Paste into ONE Kaggle cell and Run. (INDIST=True in setup below.)
# Runs: BiLSTM, MARBERT (before/after), AraBERT (before/after), FAAV-Full
# ============================================================


# --------------------------------------------------------------------
# # FAAV — Kaggle Session 2 (Two-Class: Real / Fake)
# **MARBERT before + after fine-tuning -> FAAV-Full pipeline**
# 
# ### Required Kaggle dataset
# Attach a dataset named **`faav-arabic-fake-news`** containing the 2-class submission files:
# - `claim_training_2class.csv` (~54 MB, Real/Fake only, 25,263 rows)
# - `uae_test_2class.xlsx` (balanced Real/Fake test, 5,053 rows)
# 
# (For a 3-class run instead, set `CLASS_MODE='3'` in Cell 1 and upload the full
# `claim_training.csv` + `uae_test_dataset_final.xlsx`.)
# 
# Results are saved to `/kaggle/working/results/` — download them after the session.

# -- Cell 1: Setup ------------------------------------------------------------
import os, subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'rank-bm25', 'sentence-transformers', 'openpyxl'], check=False)

CLASS_MODE = '2'                 # '2' = Real/Fake (primary) | '3' = +Unverified
INDIST     = True                # True  = in-distribution (train/test on same claim data) = HEADLINE
                                 # False = cross-domain (test on UAE-MVD) = generalization stress-test
RUN_TAG    = '_indist' if INDIST else ''
LABELS     = ['Real', 'Fake', 'Unverified'] if CLASS_MODE == '3' else ['Real', 'Fake']
print(f'CLASS_MODE={CLASS_MODE}  INDIST={INDIST}  ->  {LABELS}')

DATASET     = '/kaggle/input/faav-arabic-fake-news'
TEST_SHEET  = 'Tabelle1'
RESULTS_DIR = '/kaggle/working/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

if INDIST:
    TRAIN_FILE = os.path.join(DATASET, 'indist_train.csv')        # 80% of claim data
    TEST_FILE  = os.path.join(DATASET, 'indist_test.xlsx')        # held-out 20% (same dist)
elif CLASS_MODE == '2':
    TRAIN_FILE = os.path.join(DATASET, 'claim_training_2class.csv')
    TEST_FILE  = os.path.join(DATASET, 'uae_test_2class.xlsx')
else:
    TRAIN_FILE = os.path.join(DATASET, 'claim_training.csv')
    TEST_FILE  = os.path.join(DATASET, 'uae_test_dataset_final.xlsx')

# Fallbacks (full files filter to LABELS in code)
if not os.path.exists(TRAIN_FILE):
    alt = os.path.join(DATASET, 'claim_training.csv')
    print(f'WARNING: {os.path.basename(TRAIN_FILE)} not found -> {os.path.basename(alt)}')
    TRAIN_FILE = alt
if not os.path.exists(TEST_FILE):
    altt = os.path.join(DATASET, 'uae_test_dataset_final.xlsx')
    print(f'WARNING: {os.path.basename(TEST_FILE)} not found -> {os.path.basename(altt)}')
    TEST_FILE = altt

for f in [TRAIN_FILE, TEST_FILE]:
    ok = os.path.exists(f); mb = os.path.getsize(f)/1024**2 if ok else 0
    print(f"{'OK' if ok else 'MISSING'}  {os.path.basename(f)}  ({mb:.0f} MB)")

import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device :', DEVICE)
if torch.cuda.is_available():
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
print('Setup complete.')

# --------------------------------------------------------------------
# ---
# ## Part 0 — BiLSTM baseline
# Deep-learning baseline (claim text only). Trains fast on T4 (~3-5 min).
# Runs in whatever `CLASS_MODE` is set in Cell 1.

# ── Cell: BiLSTM baseline ─────────────────────────────────────────────────────
import os, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, cohen_kappa_score,
)

# Hyper-parameters
VOCAB_SIZE = 20_000; MAX_LEN = 200; EMBED_DIM = 128; HIDDEN_DIM = 128
NUM_LAYERS = 2; DROPOUT = 0.3; BATCH_SIZE = 32; EPOCHS = 5; LR_B = 1e-3
N_CAP = 15_000
# LABELS / CLASS_MODE / paths come from Cell 1

def _tagb(name):
    base, ext = os.path.splitext(name)
    return os.path.join(RESULTS_DIR, f'{base}_{CLASS_MODE}class{RUN_TAG}{ext}')

BILSTM_CSV     = _tagb('bilstm_claim_results.csv')
BILSTM_METRICS = _tagb('bilstm_metrics.txt')

def norm_ar(text):
    text = str(text)
    text = re.sub(r'[إأآا]', 'ا', text); text = re.sub(r'ى', 'ي', text); text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\w\s]', ' ', text); return re.sub(r'\s+', ' ', text).strip()

def build_vocab(texts, max_size):
    c = Counter(tok for t in texts for tok in t.split())
    vocab = ['<PAD>', '<UNK>'] + [w for w, _ in c.most_common(max_size - 2)]
    return {w: i for i, w in enumerate(vocab)}

def encode(text, vocab, max_len):
    ids = [vocab.get(t, 1) for t in text.split()[:max_len]]
    return ids + [0] * (max_len - len(ids))

class TextDS(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.X = [torch.tensor(encode(t, vocab, max_len), dtype=torch.long) for t in texts]
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class BiLSTM(nn.Module):
    def __init__(self, vs, ed, hd, nl, nc, dp):
        super().__init__()
        self.emb = nn.Embedding(vs, ed, padding_idx=0)
        self.lstm = nn.LSTM(ed, hd, num_layers=nl, batch_first=True,
                            bidirectional=True, dropout=dp if nl > 1 else 0)
        self.drop = nn.Dropout(dp); self.fc = nn.Linear(hd * 2, nc)
    def forward(self, x):
        e = self.drop(self.emb(x)); _, (hn, _) = self.lstm(e)
        return self.fc(self.drop(torch.cat([hn[-2], hn[-1]], dim=1)))

print(f'=== BiLSTM — {CLASS_MODE}-class {LABELS} ===')
dft = pd.read_csv(TRAIN_FILE, usecols=['text', 'label'], low_memory=False)
dft['content'] = dft['text'].fillna('').apply(norm_ar)
dft = dft[(dft['content'].str.strip() != '') & dft['label'].isin(LABELS)]
dft = pd.concat([g.sample(min(len(g), N_CAP), random_state=42) for _, g in dft.groupby('label')]).reset_index(drop=True)
le = LabelEncoder().fit(LABELS); dft['lid'] = le.transform(dft['label']); nc = len(le.classes_)
print(f'Training rows: {len(dft):,}  {dft["label"].value_counts().to_dict()}')

Xtr, Xv, ytr, yv = train_test_split(dft['content'].tolist(), dft['lid'].tolist(),
                                    test_size=0.2, random_state=42, stratify=dft['lid'].tolist())
vocab = build_vocab(Xtr, VOCAB_SIZE)
tr_dl = DataLoader(TextDS(Xtr, ytr, vocab, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)

model = BiLSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, nc, DROPOUT).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR_B); crit = nn.CrossEntropyLoss()
print(f'Training BiLSTM on {DEVICE} ...')
for ep in range(EPOCHS):
    model.train(); run = 0.0
    for xb, yb in tr_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); run += loss.item()
    print(f'  Epoch {ep+1}/{EPOCHS}  loss={run/len(tr_dl):.4f}')

# Test
dtest = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
dtest = dtest[dtest['gold_label'].isin(LABELS)].reset_index(drop=True)
dtest['cn'] = dtest['claim_ar'].fillna('').apply(norm_ar)
model.eval(); preds = []
with torch.no_grad():
    for t in dtest['cn'].tolist():
        ids = torch.tensor([encode(t, vocab, MAX_LEN)], dtype=torch.long).to(DEVICE)
        preds.append(le.classes_[int(model(ids).argmax(1).item())])
dtest['bilstm_prediction'] = preds
dtest['is_correct'] = dtest['gold_label'] == dtest['bilstm_prediction']

y_true, y_pred = dtest['gold_label'], dtest['bilstm_prediction']
acc = accuracy_score(y_true, y_pred); mf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
wf1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred); kappa = cohen_kappa_score(y_true, y_pred)
rep = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=LABELS)
print(f'\nAccuracy={acc:.4f}  MacroF1={mf1:.4f}  MCC={mcc:.4f}'); print(rep)

dtest.to_csv(BILSTM_CSV, index=False, encoding='utf-8-sig')
with open(BILSTM_METRICS, 'w', encoding='utf-8') as f:
    f.write(f'BiLSTM Baseline ({CLASS_MODE}-class)\n' + '='*40 + '\n\n')
    f.write(f'Classes  : {LABELS}\nTest rows: {len(dtest)}\n\n')
    f.write(f'Accuracy        : {acc:.4f}\nMacro F1        : {mf1:.4f}\n')
    f.write(f'Weighted F1     : {wf1:.4f}\nMCC             : {mcc:.4f}\nCohen Kappa     : {kappa:.4f}\n\n')
    f.write(rep + '\nConfusion Matrix:\n' + str(cm))
print(f'\nSaved -> {BILSTM_CSV}')

# --------------------------------------------------------------------
# ---
# ## Part 1 — MARBERT
# - Phase 1: evaluate pre-trained model (before fine-tuning, random head)
# - Phase 2: fine-tune on 120k training rows (3 epochs)
# - Phase 3: evaluate fine-tuned model
# 
# Estimated time on T4: **~2.5–3 hours**

# ── Cell 2: MARBERT ───────────────────────────────────────────────────────────
import os, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = 'UBC-NLP/MARBERTv2'
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5
# LABELS and CLASS_MODE come from Cell 1

def _tag(name):
    base, ext = os.path.splitext(name)
    return os.path.join(RESULTS_DIR, f'{base}_{CLASS_MODE}class{RUN_TAG}{ext}')

MARBERT_BEFORE_CSV     = _tag('marbert_before_finetune_results.csv')
MARBERT_BEFORE_METRICS = _tag('marbert_before_finetune_metrics.txt')
MARBERT_CSV            = _tag('marbert_claim_results.csv')
MARBERT_METRICS        = _tag('marbert_metrics.txt')
MARBERT_SAVE           = os.path.join(RESULTS_DIR, f'marbert_finetuned_{CLASS_MODE}class{RUN_TAG}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_arabic(text):
    text = str(text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

class ArabicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, evidences=None):
        self.texts = texts; self.evidences = evidences
        self.labels = labels; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        ev = self.evidences[idx] if self.evidences else None
        enc = self.tokenizer(self.texts[idx], ev, max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def compute_metrics(y_true, y_pred, y_proba):
    acc   = accuracy_score(y_true, y_pred)
    mf1   = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    wf1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    rep   = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred, labels=LABELS)
    try:
        auc = roc_auc_score(pd.get_dummies(y_true).values, y_proba,
                            multi_class='ovr', average='macro')
    except Exception: auc = float('nan')
    return dict(acc=acc, macro_f1=mf1, weighted_f1=wf1, mcc=mcc, kappa=kappa, auc=auc, report=rep, matrix=cm)

def save_metrics(path, title, m, phase, n_train):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'{title} ({CLASS_MODE}-class)\n{"="*len(title)}\n\n')
        f.write(f'Model     : {MODEL_NAME}\nDevice    : {DEVICE}\n')
        f.write(f'Classes   : {LABELS}\nPhase     : {phase}\nTrain rows: {n_train}\n\n')
        f.write(f'Accuracy        : {m["acc"]:.4f}\n')
        f.write(f'Macro F1        : {m["macro_f1"]:.4f}\n')
        f.write(f'Weighted F1     : {m["weighted_f1"]:.4f}\n')
        f.write(f'MCC             : {m["mcc"]:.4f}\n')
        f.write(f'Cohen Kappa     : {m["kappa"]:.4f}\n')
        f.write(f'AUC-ROC (macro) : {m["auc"]:.4f}\n\n')
        f.write('Classification Report:\n'); f.write(m['report'])
        f.write('\nConfusion Matrix:\n'); f.write(str(m['matrix']))

def run_inference(model, tokenizer, claims, evidences, le):
    model.eval()
    all_preds, all_proba = [], []
    with torch.no_grad():
        for i in range(0, len(claims), BATCH_SIZE):
            enc = tokenizer(claims[i:i+BATCH_SIZE], evidences[i:i+BATCH_SIZE],
                            max_length=MAX_LEN, padding='max_length',
                            truncation=True, return_tensors='pt')
            logits = model(input_ids=enc['input_ids'].to(DEVICE),
                           attention_mask=enc['attention_mask'].to(DEVICE)).logits
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            all_proba.extend(proba.tolist())
            all_preds.extend(le.inverse_transform(np.argmax(proba, axis=1)))
    return all_preds, np.array(all_proba)

# ── Load data (claim + evidence, filtered to LABELS) ──────────────────────────
print(f'Loading training data ... (CLASS_MODE={CLASS_MODE})')
_cols = pd.read_csv(TRAIN_FILE, nrows=1).columns
_use  = [c for c in ['text', 'label', 'evidence_text'] if c in _cols]
df_train = pd.read_csv(TRAIN_FILE, usecols=_use, low_memory=False)
df_train = df_train[df_train['label'].isin(LABELS)].dropna(subset=['text', 'label'])
df_train['text'] = df_train['text'].apply(normalize_arabic)
if 'evidence_text' in df_train.columns:
    df_train['evidence_text'] = df_train['evidence_text'].fillna('').apply(normalize_arabic)
else:
    df_train['evidence_text'] = ''
print(f'Training rows: {len(df_train):,}  {df_train["label"].value_counts().to_dict()}')

le = LabelEncoder().fit(LABELS)
df_train['label_id'] = le.transform(df_train['label'])
num_classes = len(le.classes_)

print('Loading test data ...')
df_test     = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
df_test     = df_test[df_test['gold_label'].isin(LABELS)].reset_index(drop=True)
test_claims = [normalize_arabic(str(c)) for c in df_test['claim_ar']]
test_evs    = [normalize_arabic(str(e)) for e in df_test['evidence_text'].fillna('')]
print(f'Test rows: {len(df_test):,}  {df_test["gold_label"].value_counts().to_dict()}')

print(f'\nLoading tokenizer: {MODEL_NAME} ...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Phase 1: Before fine-tuning ───────────────────────────────────────────────
print('\n--- PHASE 1: Before fine-tuning ---')
model_before = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True).to(DEVICE)
preds_before, proba_before = run_inference(model_before, tokenizer, test_claims, test_evs, le)
m_before = compute_metrics(df_test['gold_label'].tolist(), preds_before, proba_before)
print(f'Accuracy: {m_before["acc"]:.4f}  Macro F1: {m_before["macro_f1"]:.4f}')
df_b = df_test.copy()
df_b['marbert_prediction'] = preds_before
df_b['is_correct'] = df_b['gold_label'] == df_b['marbert_prediction']
df_b.to_csv(MARBERT_BEFORE_CSV, index=False, encoding='utf-8-sig')
save_metrics(MARBERT_BEFORE_METRICS, 'MARBERT Before Fine-tuning', m_before,
             'Before fine-tuning (random head)', len(df_train))
del model_before
torch.cuda.empty_cache()

# ── Phase 2: Fine-tuning ──────────────────────────────────────────────────────
print('\n--- PHASE 2: Fine-tuning (3 epochs, claim + evidence) ---')
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True).to(DEVICE)
train_dl = DataLoader(
    ArabicDataset(df_train['text'].tolist(), df_train['label_id'].tolist(), tokenizer, MAX_LEN,
                  evidences=df_train['evidence_text'].tolist()),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
counts   = Counter(df_train['label_id'].tolist())
total_n  = sum(counts.values())
class_w  = torch.tensor([total_n / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
                         dtype=torch.float, device=DEVICE)
loss_fn   = nn.CrossEntropyLoss(weight=class_w)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
epoch_losses = []
for epoch in range(EPOCHS):
    model.train(); running = 0.0
    for batch in train_dl:
        ids = batch['input_ids'].to(DEVICE); mask = batch['attention_mask'].to(DEVICE)
        labs = batch['label'].to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); running += loss.item()
    avg = running / len(train_dl); epoch_losses.append(avg)
    print(f'  Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}')
pd.DataFrame({'epoch': range(1, EPOCHS+1), 'loss': epoch_losses}).to_csv(
    _tag('marbert_training_loss.csv'), index=False)
os.makedirs(MARBERT_SAVE, exist_ok=True)
model.save_pretrained(MARBERT_SAVE); tokenizer.save_pretrained(MARBERT_SAVE)
print(f'Model saved -> {MARBERT_SAVE}')

# ── Phase 3: After fine-tuning ────────────────────────────────────────────────
print('\n--- PHASE 3: After fine-tuning ---')
preds_after, proba_after = run_inference(model, tokenizer, test_claims, test_evs, le)
m_after = compute_metrics(df_test['gold_label'].tolist(), preds_after, proba_after)
print(f'Accuracy: {m_after["acc"]:.4f}  Macro F1: {m_after["macro_f1"]:.4f}')
print(m_after['report'])
df_a = df_test.copy()
df_a['marbert_prediction'] = preds_after
for lbl in LABELS:
    df_a[f'marbert_proba_{lbl.lower()}'] = proba_after[:, le.transform([lbl])[0]]
df_a['is_correct'] = df_a['gold_label'] == df_a['marbert_prediction']
df_a.to_csv(MARBERT_CSV, index=False, encoding='utf-8-sig')
save_metrics(MARBERT_METRICS, 'MARBERT After Fine-tuning', m_after, 'After fine-tuning', len(df_train))

print(f'\n=== MARBERT SUMMARY ({CLASS_MODE}-class) ===')
print(f'{"Metric":<20} {"Before":>10} {"After":>10}')
print('-' * 42)
for k, lbl in [("acc","Accuracy"),("macro_f1","Macro F1"),("mcc","MCC"),("kappa","Kappa")]:
    print(f'{lbl:<20} {m_before[k]:>10.4f} {m_after[k]:>10.4f}')
print(f'\nSaved: {MARBERT_CSV}')

# --------------------------------------------------------------------
# ---
# ## Part 2b - AraBERT (fine-tuned)
# Fine-tunes AraBERT on claim+evidence, before/after, like MARBERT.

# ── Cell 2: AraBERT ───────────────────────────────────────────────────────────
import os, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = 'aubmindlab/bert-base-arabertv02'
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5
# LABELS and CLASS_MODE come from Cell 1

def _tag(name):
    base, ext = os.path.splitext(name)
    return os.path.join(RESULTS_DIR, f'{base}_{CLASS_MODE}class{RUN_TAG}{ext}')

AraBERT_BEFORE_CSV     = _tag('arabert_before_finetune_results.csv')
AraBERT_BEFORE_METRICS = _tag('arabert_before_finetune_metrics.txt')
AraBERT_CSV            = _tag('arabert_claim_results.csv')
AraBERT_METRICS        = _tag('arabert_metrics.txt')
AraBERT_SAVE           = os.path.join(RESULTS_DIR, f'arabert_finetuned_{CLASS_MODE}class{RUN_TAG}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_arabic(text):
    text = str(text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

class ArabicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, evidences=None):
        self.texts = texts; self.evidences = evidences
        self.labels = labels; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        ev = self.evidences[idx] if self.evidences else None
        enc = self.tokenizer(self.texts[idx], ev, max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def compute_metrics(y_true, y_pred, y_proba):
    acc   = accuracy_score(y_true, y_pred)
    mf1   = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    wf1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    rep   = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred, labels=LABELS)
    try:
        auc = roc_auc_score(pd.get_dummies(y_true).values, y_proba,
                            multi_class='ovr', average='macro')
    except Exception: auc = float('nan')
    return dict(acc=acc, macro_f1=mf1, weighted_f1=wf1, mcc=mcc, kappa=kappa, auc=auc, report=rep, matrix=cm)

def save_metrics(path, title, m, phase, n_train):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'{title} ({CLASS_MODE}-class)\n{"="*len(title)}\n\n')
        f.write(f'Model     : {MODEL_NAME}\nDevice    : {DEVICE}\n')
        f.write(f'Classes   : {LABELS}\nPhase     : {phase}\nTrain rows: {n_train}\n\n')
        f.write(f'Accuracy        : {m["acc"]:.4f}\n')
        f.write(f'Macro F1        : {m["macro_f1"]:.4f}\n')
        f.write(f'Weighted F1     : {m["weighted_f1"]:.4f}\n')
        f.write(f'MCC             : {m["mcc"]:.4f}\n')
        f.write(f'Cohen Kappa     : {m["kappa"]:.4f}\n')
        f.write(f'AUC-ROC (macro) : {m["auc"]:.4f}\n\n')
        f.write('Classification Report:\n'); f.write(m['report'])
        f.write('\nConfusion Matrix:\n'); f.write(str(m['matrix']))

def run_inference(model, tokenizer, claims, evidences, le):
    model.eval()
    all_preds, all_proba = [], []
    with torch.no_grad():
        for i in range(0, len(claims), BATCH_SIZE):
            enc = tokenizer(claims[i:i+BATCH_SIZE], evidences[i:i+BATCH_SIZE],
                            max_length=MAX_LEN, padding='max_length',
                            truncation=True, return_tensors='pt')
            logits = model(input_ids=enc['input_ids'].to(DEVICE),
                           attention_mask=enc['attention_mask'].to(DEVICE)).logits
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            all_proba.extend(proba.tolist())
            all_preds.extend(le.inverse_transform(np.argmax(proba, axis=1)))
    return all_preds, np.array(all_proba)

# ── Load data (claim + evidence, filtered to LABELS) ──────────────────────────
print(f'Loading training data ... (CLASS_MODE={CLASS_MODE})')
_cols = pd.read_csv(TRAIN_FILE, nrows=1).columns
_use  = [c for c in ['text', 'label', 'evidence_text'] if c in _cols]
df_train = pd.read_csv(TRAIN_FILE, usecols=_use, low_memory=False)
df_train = df_train[df_train['label'].isin(LABELS)].dropna(subset=['text', 'label'])
df_train['text'] = df_train['text'].apply(normalize_arabic)
if 'evidence_text' in df_train.columns:
    df_train['evidence_text'] = df_train['evidence_text'].fillna('').apply(normalize_arabic)
else:
    df_train['evidence_text'] = ''
print(f'Training rows: {len(df_train):,}  {df_train["label"].value_counts().to_dict()}')

le = LabelEncoder().fit(LABELS)
df_train['label_id'] = le.transform(df_train['label'])
num_classes = len(le.classes_)

print('Loading test data ...')
df_test     = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
df_test     = df_test[df_test['gold_label'].isin(LABELS)].reset_index(drop=True)
test_claims = [normalize_arabic(str(c)) for c in df_test['claim_ar']]
test_evs    = [normalize_arabic(str(e)) for e in df_test['evidence_text'].fillna('')]
print(f'Test rows: {len(df_test):,}  {df_test["gold_label"].value_counts().to_dict()}')

print(f'\nLoading tokenizer: {MODEL_NAME} ...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Phase 1: Before fine-tuning ───────────────────────────────────────────────
print('\n--- PHASE 1: Before fine-tuning ---')
model_before = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True).to(DEVICE)
preds_before, proba_before = run_inference(model_before, tokenizer, test_claims, test_evs, le)
m_before = compute_metrics(df_test['gold_label'].tolist(), preds_before, proba_before)
print(f'Accuracy: {m_before["acc"]:.4f}  Macro F1: {m_before["macro_f1"]:.4f}')
df_b = df_test.copy()
df_b['arabert_prediction'] = preds_before
df_b['is_correct'] = df_b['gold_label'] == df_b['arabert_prediction']
df_b.to_csv(AraBERT_BEFORE_CSV, index=False, encoding='utf-8-sig')
save_metrics(AraBERT_BEFORE_METRICS, 'AraBERT Before Fine-tuning', m_before,
             'Before fine-tuning (random head)', len(df_train))
del model_before
torch.cuda.empty_cache()

# ── Phase 2: Fine-tuning ──────────────────────────────────────────────────────
print('\n--- PHASE 2: Fine-tuning (3 epochs, claim + evidence) ---')
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_classes, ignore_mismatched_sizes=True).to(DEVICE)
train_dl = DataLoader(
    ArabicDataset(df_train['text'].tolist(), df_train['label_id'].tolist(), tokenizer, MAX_LEN,
                  evidences=df_train['evidence_text'].tolist()),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
counts   = Counter(df_train['label_id'].tolist())
total_n  = sum(counts.values())
class_w  = torch.tensor([total_n / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
                         dtype=torch.float, device=DEVICE)
loss_fn   = nn.CrossEntropyLoss(weight=class_w)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
epoch_losses = []
for epoch in range(EPOCHS):
    model.train(); running = 0.0
    for batch in train_dl:
        ids = batch['input_ids'].to(DEVICE); mask = batch['attention_mask'].to(DEVICE)
        labs = batch['label'].to(DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); running += loss.item()
    avg = running / len(train_dl); epoch_losses.append(avg)
    print(f'  Epoch {epoch+1}/{EPOCHS}  loss={avg:.4f}')
pd.DataFrame({'epoch': range(1, EPOCHS+1), 'loss': epoch_losses}).to_csv(
    _tag('arabert_training_loss.csv'), index=False)
os.makedirs(AraBERT_SAVE, exist_ok=True)
model.save_pretrained(AraBERT_SAVE); tokenizer.save_pretrained(AraBERT_SAVE)
print(f'Model saved -> {AraBERT_SAVE}')

# ── Phase 3: After fine-tuning ────────────────────────────────────────────────
print('\n--- PHASE 3: After fine-tuning ---')
preds_after, proba_after = run_inference(model, tokenizer, test_claims, test_evs, le)
m_after = compute_metrics(df_test['gold_label'].tolist(), preds_after, proba_after)
print(f'Accuracy: {m_after["acc"]:.4f}  Macro F1: {m_after["macro_f1"]:.4f}')
print(m_after['report'])
df_a = df_test.copy()
df_a['arabert_prediction'] = preds_after
for lbl in LABELS:
    df_a[f'arabert_proba_{lbl.lower()}'] = proba_after[:, le.transform([lbl])[0]]
df_a['is_correct'] = df_a['gold_label'] == df_a['arabert_prediction']
df_a.to_csv(AraBERT_CSV, index=False, encoding='utf-8-sig')
save_metrics(AraBERT_METRICS, 'AraBERT After Fine-tuning', m_after, 'After fine-tuning', len(df_train))

print(f'\n=== AraBERT SUMMARY ({CLASS_MODE}-class) ===')
print(f'{"Metric":<20} {"Before":>10} {"After":>10}')
print('-' * 42)
for k, lbl in [("acc","Accuracy"),("macro_f1","Macro F1"),("mcc","MCC"),("kappa","Kappa")]:
    print(f'{lbl:<20} {m_before[k]:>10.4f} {m_after[k]:>10.4f}')
print(f'\nSaved: {AraBERT_CSV}')

# --------------------------------------------------------------------
# ---
# ## Part 2 — FAAV-Full
# BM25 retrieval → MARBERT NLI → MiniLM faithfulness scoring
# 
# Estimated time on T4: **~2–3 hours**

# ── Cell 3: FAAV-Full ─────────────────────────────────────────────────────────
import os, re, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
)
try:
    from rank_bm25 import BM25Okapi; HAS_BM25 = True
except ImportError:
    HAS_BM25 = False; print('rank_bm25 not found — pip install rank-bm25')
try:
    from sentence_transformers import SentenceTransformer, util as st_util; HAS_SBERT = True
except ImportError:
    HAS_SBERT = False; print('sentence-transformers not found')

# ── Config ────────────────────────────────────────────────────────────────────
NLI_BASE      = 'UBC-NLP/MARBERTv2'
SENT_MODEL    = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MAX_LEN       = 128
BATCH_SIZE    = 32
EPOCHS        = 3
LR            = 2e-5
CORPUS_LIMIT  = 50_000
TOP_K         = 5
FAITH_THRESH  = 0.30
# LABELS and CLASS_MODE come from Cell 1

def _tag(name):
    base, ext = os.path.splitext(name)
    return os.path.join(RESULTS_DIR, f'{base}_{CLASS_MODE}class{RUN_TAG}{ext}')

NLI_MODEL_DIR = os.path.join(RESULTS_DIR, f'faav_nli_model_{CLASS_MODE}class{RUN_TAG}')
FAAV_CSV      = _tag('faav_full_results.csv')
FAAV_METRICS  = _tag('faav_full_metrics.txt')

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_arabic(text):
    text = str(text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

class NLIDataset(Dataset):
    def __init__(self, claims, evidences, labels, tokenizer, max_len):
        self.claims = claims; self.evidences = evidences
        self.labels = labels; self.tokenizer = tokenizer; self.max_len = max_len
    def __len__(self): return len(self.claims)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.claims[idx], self.evidences[idx], max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def train_nli(save_dir):
    print('Loading training data for NLI ...')
    _cols = pd.read_csv(TRAIN_FILE, nrows=1).columns
    _use  = [c for c in ['text', 'label', 'evidence_text'] if c in _cols]
    df = pd.read_csv(TRAIN_FILE, usecols=_use, low_memory=False)
    df = df[df['label'].isin(LABELS)].dropna(subset=['text', 'label'])
    if 'evidence_text' not in df.columns:
        df['evidence_text'] = ''
    df['evidence_text'] = df['evidence_text'].fillna('')
    df['c'] = df['text'].apply(normalize_arabic)
    df['e'] = df['evidence_text'].apply(normalize_arabic)
    le = LabelEncoder().fit(LABELS)
    df['lid'] = le.transform(df['label'])
    nc = len(le.classes_)
    print(f'NLI training: {len(df):,} rows  {df["label"].value_counts().to_dict()}')
    tok = AutoTokenizer.from_pretrained(NLI_BASE)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        NLI_BASE, num_labels=nc, ignore_mismatched_sizes=True).to(DEVICE)
    dl = DataLoader(NLIDataset(df['c'].tolist(), df['e'].tolist(),
                               df['lid'].tolist(), tok, MAX_LEN),
                    batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    counts = Counter(df['lid'].tolist()); tot = sum(counts.values())
    cw = torch.tensor([tot/(nc*counts.get(i,1)) for i in range(nc)],
                       dtype=torch.float, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw)
    opt = AdamW(mdl.parameters(), lr=LR, weight_decay=0.01)
    ts = len(dl)*EPOCHS
    sched = get_linear_schedule_with_warmup(opt, max(1,ts//10), ts)
    print(f'Fine-tuning NLI on {DEVICE} ...')
    for ep in range(EPOCHS):
        mdl.train(); run = 0.0
        for b in dl:
            ids=b['input_ids'].to(DEVICE); mask=b['attention_mask'].to(DEVICE)
            labs=b['label'].to(DEVICE)
            opt.zero_grad()
            loss=loss_fn(mdl(input_ids=ids,attention_mask=mask).logits,labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
            opt.step(); sched.step(); run+=loss.item()
        print(f'  Epoch {ep+1}/{EPOCHS}  loss={run/len(dl):.4f}')
    os.makedirs(save_dir, exist_ok=True)
    mdl.save_pretrained(save_dir); tok.save_pretrained(save_dir)
    with open(os.path.join(save_dir,'label_classes.json'),'w') as f:
        json.dump(le.classes_.tolist(), f)
    print(f'NLI model saved -> {save_dir}')
    return mdl, tok, le

# ── Load or train NLI ─────────────────────────────────────────────────────────
lc_path = os.path.join(NLI_MODEL_DIR, 'label_classes.json')
if os.path.exists(lc_path):
    print(f'Loading saved NLI model from {NLI_MODEL_DIR} ...')
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_DIR)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_DIR).to(DEVICE)
    le = LabelEncoder(); le.fit(json.load(open(lc_path)))
else:
    nli_model, tokenizer, le = train_nli(NLI_MODEL_DIR)
nli_model.eval()
num_classes = len(le.classes_)
fake_idx = le.transform(['Fake'])[0]

# ── BM25 index ────────────────────────────────────────────────────────────────
bm25 = None; corpus_texts = []
if HAS_BM25:
    print(f'\nBuilding BM25 index (cap={CORPUS_LIMIT:,}) ...')
    df_corp = pd.read_csv(TRAIN_FILE, usecols=['text'], low_memory=False).dropna()
    corpus_texts = df_corp['text'].head(CORPUS_LIMIT).tolist()
    bm25 = BM25Okapi([normalize_arabic(t).split() for t in corpus_texts])
    print(f'BM25 ready — {len(corpus_texts):,} docs')

# ── SentenceTransformer ───────────────────────────────────────────────────────
sent_model = SentenceTransformer(SENT_MODEL) if HAS_SBERT else None
print('SentenceTransformer loaded.' if sent_model else 'Faithfulness: skipped')

# ── Load test ─────────────────────────────────────────────────────────────────
print(f'\nLoading test set ... (CLASS_MODE={CLASS_MODE})')
df_test     = pd.read_excel(TEST_FILE, sheet_name=TEST_SHEET)
df_test     = df_test[df_test['gold_label'].isin(LABELS)].reset_index(drop=True)
claims_norm = [normalize_arabic(str(c)) for c in df_test['claim_ar']]
evs_norm    = [normalize_arabic(str(e)) for e in df_test['evidence_text'].fillna('')]
print(f'{len(df_test):,} claims  {df_test["gold_label"].value_counts().to_dict()}')

# ── BM25 retrieval ────────────────────────────────────────────────────────────
augmented_ev = []; bm25_scores = []
for cn, ex in zip(claims_norm, evs_norm):
    if bm25 is not None:
        scores = bm25.get_scores(cn.split())
        top_idx = np.argsort(scores)[::-1][:TOP_K]
        retrieved = ' '.join(corpus_texts[j] for j in top_idx)
        combined  = (ex + ' ' + retrieved).strip() if ex else retrieved
        bm25_scores.append(round(float(scores[top_idx[0]]) if len(top_idx) else 0.0, 4))
    else:
        combined = ex; bm25_scores.append(0.0)
    augmented_ev.append(combined)

# ── NLI inference ─────────────────────────────────────────────────────────────
print('Running NLI inference ...')
nli_preds = []; contra_scores = []; all_proba = []
with torch.no_grad():
    for i in range(0, len(claims_norm), BATCH_SIZE):
        enc = tokenizer(claims_norm[i:i+BATCH_SIZE], augmented_ev[i:i+BATCH_SIZE],
                        max_length=MAX_LEN, padding='max_length', truncation=True, return_tensors='pt')
        logits = nli_model(input_ids=enc['input_ids'].to(DEVICE),
                           attention_mask=enc['attention_mask'].to(DEVICE)).logits
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        nli_preds.extend(le.inverse_transform(np.argmax(proba, axis=1)))
        contra_scores.extend(proba[:, fake_idx].tolist())
        all_proba.extend(proba.tolist())
print(f'NLI done. {Counter(nli_preds)}')

# ── Faithfulness + results ────────────────────────────────────────────────────
rows = []
for idx, row in df_test.iterrows():
    pred = nli_preds[idx]
    expl = ('The evidence contradicts the claim.' if pred=='Fake' else
            'Evidence is ambiguous.' if pred=='Unverified' else
            'The evidence supports the claim.')
    if sent_model:
        c_e = sent_model.encode(claims_norm[idx], convert_to_tensor=True)
        v_e = sent_model.encode(augmented_ev[idx][:512], convert_to_tensor=True)
        x_e = sent_model.encode(expl, convert_to_tensor=True)
        sem  = float(st_util.cos_sim(c_e, v_e).item())
        grd  = float(st_util.cos_sim(x_e, v_e).item())
        faith = 'High' if grd >= FAITH_THRESH else 'Low'
    else:
        sem = grd = 0.0; faith = 'High'
    rows.append({'claim_ar': row['claim_ar'], 'gold_label': row['gold_label'],
                 'faav_prediction': pred, 'explanation': expl,
                 'bm25_top_score': bm25_scores[idx],
                 'semantic_similarity': round(sem,4),
                 'contradiction_score': round(contra_scores[idx],4),
                 'grounding_score': round(grd,4),
                 'predicted_faithfulness': faith,
                 'is_correct': pred == row['gold_label']})

df_out = pd.DataFrame(rows)

# ── Classification metrics ─────────────────────────────────────────────────────
y_true = df_out['gold_label'].tolist(); y_pred = df_out['faav_prediction'].tolist()
acc   = accuracy_score(y_true, y_pred)
prec  = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec   = recall_score(y_true, y_pred, average='macro', zero_division=0)
mf1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
wf1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
mcc   = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
rep   = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
cm    = confusion_matrix(y_true, y_pred, labels=LABELS)
try:
    auc = roc_auc_score(pd.get_dummies(y_true).values, np.array(all_proba),
                        multi_class='ovr', average='macro')
except Exception: auc = float('nan')

faith_acc   = accuracy_score(df_test['faithfulness_label'], df_out['predicted_faithfulness'])
halluc      = (df_out['predicted_faithfulness']=='Low').mean()
avg_sem     = df_out['semantic_similarity'].mean()
avg_grd     = df_out['grounding_score'].mean()
avg_contra  = df_out['contradiction_score'].mean()
if 'retrieval_quality' in df_test.columns:
    rq = df_test['retrieval_quality'].value_counts(normalize=True)*100
    ev_rel = (rq.get('Relevant',0)*1.0 + rq.get('Partially Relevant',0)*0.5)/100.0
else:
    ev_rel = float('nan')

print(f'\n=== FAAV-Full Results ({CLASS_MODE}-class) ===')
print(f'Accuracy        : {acc:.4f}'); print(f'Macro F1        : {mf1:.4f}')
print(f'MCC             : {mcc:.4f}'); print(f'Kappa           : {kappa:.4f}')
print(f'AUC-ROC         : {auc:.4f}')
print(f'Faithfulness Acc: {faith_acc:.4f}'); print(f'Hallucination   : {halluc:.4f}')
print(f'\n{rep}'); print(f'Confusion Matrix:\n{cm}')

df_out.to_csv(FAAV_CSV, index=False, encoding='utf-8-sig')
with open(FAAV_METRICS, 'w', encoding='utf-8') as f:
    f.write(f'FAAV-Full Pipeline Results ({CLASS_MODE}-class)\n==========================\n\n')
    f.write(f'Classes     : {LABELS}\n')
    f.write(f'Retriever   : BM25Okapi (corpus={len(corpus_texts):,}, top-k={TOP_K})\n')
    f.write(f'NLI model   : {NLI_BASE} fine-tuned\nDevice      : {DEVICE}\n\n')
    f.write(f'Accuracy        : {acc:.4f}\nMacro Precision : {prec:.4f}\n')
    f.write(f'Macro Recall    : {rec:.4f}\nMacro F1        : {mf1:.4f}\n')
    f.write(f'Weighted F1     : {wf1:.4f}\nMCC             : {mcc:.4f}\n')
    f.write(f'Cohen Kappa     : {kappa:.4f}\nAUC-ROC (macro) : {auc:.4f}\n\n')
    f.write(f'Classification Report:\n{rep}\nConfusion Matrix:\n{cm}\n\n')
    f.write(f'Evidence Relevance Score  : {ev_rel:.4f}\n')
    f.write(f'Faithfulness Accuracy     : {faith_acc:.4f}\n')
    f.write(f'Hallucination Rate        : {halluc:.4f}\n')
    f.write(f'Avg Semantic Similarity   : {avg_sem:.4f}\n')
    f.write(f'Avg Contradiction Score   : {avg_contra:.4f}\n')
    f.write(f'Avg Evidence Grounding    : {avg_grd:.4f}\n')

print(f'\nResults -> {FAAV_CSV}')
print(f'Metrics -> {FAAV_METRICS}')
print('\nDownload everything from /kaggle/working/results/')