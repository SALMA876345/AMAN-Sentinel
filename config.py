"""
Centralised path configuration for the FAAV project.
All scripts import from here — change paths in ONE place only.
"""
import os

# Project root — works both as a script file and inside a Kaggle notebook cell
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()   # Kaggle: notebook cell has no __file__

DATA_DIR    = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

# Reproducibility and research toggles. These are environment-overridable so
# old scripts keep their defaults while cross-domain runs can be configured
# from the command line.
SEED = int(os.environ.get("SEED", "42"))
MODEL_INPUT_MODE = os.environ.get("MODEL_INPUT_MODE", "claim_only").strip()
if MODEL_INPUT_MODE not in {"claim_only", "claim_plus_evidence"}:
    raise ValueError("MODEL_INPUT_MODE must be 'claim_only' or 'claim_plus_evidence'")

RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "bm25").strip()
if RETRIEVAL_MODE not in {"none", "bm25", "tfidf", "dense", "hybrid"}:
    raise ValueError(
        "RETRIEVAL_MODE must be one of: none, bm25, tfidf, dense, hybrid"
    )

TOP_K_EVIDENCE = int(os.environ.get("TOP_K_EVIDENCE", "5"))
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", "0.50"))
DENSE_WEIGHT = float(os.environ.get("DENSE_WEIGHT", "0.50"))
DOMAIN_BALANCED = os.environ.get("DOMAIN_BALANCED", "0").strip() == "1"
FEWSHOT_K = int(os.environ.get("FEWSHOT_K", "0"))

# ── Input files ───────────────────────────────────────────────────────────────
UNIFIED_TRAIN_FILE       = os.path.join(DATA_DIR, "unified_training.csv")          # all 5 datasets (legacy)
TWEET_TRAIN_FILE         = os.path.join(DATA_DIR, "tweet_training.csv")            # VERA-ARAB + ArCOV19 tweets only
FOUR_DATASET_TRAIN_FILE  = os.path.join(DATA_DIR, "four_dataset_training.csv")     # AFND + ArCOV19 + VERA-ARAB + AraStance (legacy — source-credibility mix)
CLAIM_TRAIN_FILE         = os.path.join(DATA_DIR, "claim_training.csv")            # claim-truth aligned: VERA+AraStance+ArCOV (Real/Fake) + AFND (Unverified) + evidence
CLAIM_TRAIN_2CLASS_FILE  = os.path.join(DATA_DIR, "claim_training_2class.csv")      # Real/Fake only (no AFND/Unverified) — submission file
RELEVANCE_TRAIN_FILE     = os.path.join(DATA_DIR, "relevance_training.csv")         # FAAV gate: claim+evidence -> usable/not_usable
STANCE_TRAIN_FILE        = os.path.join(DATA_DIR, "stance_training.csv")            # FAAV verifier: claim+usable evidence -> Supported/Refuted
TRAIN_FILE               = CLAIM_TRAIN_FILE   # claim-truth aligned to match UAE-MVD test task
TEST_FILE          = os.path.join(DATA_DIR, "uae_test_dataset_final.xlsx")  # UAE-MVD 30,015 rows (standardised)
TEST_2CLASS_FILE   = os.path.join(DATA_DIR, "uae_test_2class.xlsx")          # balanced Real/Fake test (5,053) — submission file
TEST_SHEET         = "Tabelle1"
MANUAL_TEST_FILE   = os.path.join(DATA_DIR, "manual_claim_test_dataset.xlsx")  # original 15-claim set
ANS_DIR            = os.path.join(DATA_DIR, "ANS")

# ── Class mode (2-class Real/Fake  or  3-class Real/Fake/Unverified) ──────────
# Override per-run with env var:  CLASS_MODE=2  python baselines/baseline_svm.py
CLASS_MODE = os.environ.get("CLASS_MODE", "3").strip()
if CLASS_MODE == "2":
    CLASS_LABELS = ["Real", "Fake"]
    # In 2-class mode read the dedicated Real/Fake-only files (submission set)
    if os.path.exists(CLAIM_TRAIN_2CLASS_FILE):
        TRAIN_FILE = CLAIM_TRAIN_2CLASS_FILE
    if os.path.exists(TEST_2CLASS_FILE):
        TEST_FILE = TEST_2CLASS_FILE
else:
    CLASS_MODE = "3"
    CLASS_LABELS = ["Real", "Fake", "Unverified"]

# ── In-distribution mode: INDIST=1 trains/tests on an 80/20 split of the claim
#    data itself (matched distribution), instead of cross-domain UAE-MVD. ───────
INDIST = os.environ.get("INDIST", "0").strip() == "1"
INDIST_TRAIN_FILE = os.path.join(DATA_DIR, "indist_train.csv")
INDIST_TEST_FILE  = os.path.join(DATA_DIR, "indist_test.xlsx")
if INDIST:
    if os.path.exists(INDIST_TRAIN_FILE):
        TRAIN_FILE = INDIST_TRAIN_FILE
    if os.path.exists(INDIST_TEST_FILE):
        TEST_FILE = INDIST_TEST_FILE

def result_tagged(filename):
    """Insert the class-mode tag before the extension, e.g. svm_metrics_3class.txt
    (and _indist when running the in-distribution split)."""
    base, ext = os.path.splitext(filename)
    suffix = f"_{CLASS_MODE}class" + ("_indist" if INDIST else "")
    return os.path.join(RESULTS_DIR, f"{base}{suffix}{ext}")

# ── AFND corpus (for BM25 / prepare scripts) ─────────────────────────────────
AFND_DIR       = os.path.join(ROOT, "..", "AFND_Selected")
SOURCES_FILE   = os.path.join(AFND_DIR, "sources.json")

# ── Result files ──────────────────────────────────────────────────────────────
def result(filename):
    """Return full path to a file inside results/."""
    return os.path.join(RESULTS_DIR, filename)

# Baseline outputs
SVM_CSV         = result("svm_claim_results.csv")
SVM_METRICS     = result("svm_metrics.txt")
LR_CSV          = result("lr_claim_results.csv")
LR_METRICS      = result("lr_claim_metrics.txt")
BILSTM_CSV      = result("bilstm_claim_results.csv")
BILSTM_METRICS  = result("bilstm_metrics.txt")
ARABERT_CSV     = result("arabert_claim_results.csv")
ARABERT_METRICS = result("arabert_metrics.txt")
ARABERT_SAVE    = result("arabert_finetuned")

# RAG / FAAV outputs
STD_RAG_CSV     = result("standard_rag_results.csv")
STD_RAG_METRICS = result("standard_rag_metrics.txt")
FAAV_LIGHT_CSV  = result("faithfulness_aware_rag_results.csv")
FAAV_LIGHT_METRICS = result("faithfulness_aware_rag_metrics.txt")
FAAV_FULL_CSV   = result("faav_full_results.csv")
FAAV_FULL_METRICS = result("faav_full_metrics.txt")

# MARBERT outputs
MARBERT_CSV          = result("marbert_claim_results.csv")
MARBERT_METRICS      = result("marbert_metrics.txt")
MARBERT_SAVE         = result("marbert_finetuned")
MARBERT_BEFORE_CSV   = result("marbert_before_finetune_results.csv")
MARBERT_BEFORE_METRICS = result("marbert_before_finetune_metrics.txt")

# AraBERT outputs (updated)
ARABERT_BEFORE_CSV     = result("arabert_before_finetune_results.csv")
ARABERT_BEFORE_METRICS = result("arabert_before_finetune_metrics.txt")

# Cross-validation outputs
CV_METRICS      = result("cross_validation_metrics.txt")
CV_CSV          = result("cross_validation_results.csv")

# Ablation outputs
ABLATION_CSV    = result("ablation_results.csv")
ABLATION_METRICS = result("ablation_metrics.txt")

# FAAV full metrics
FAAV_FULL_CSV     = result("faav_full_results.csv")
FAAV_FULL_METRICS = result("faav_full_metrics.txt")

# Merged test file
MERGED_TEST_FILE  = result("merged_test_dataset.csv")

# Graph output directory
GRAPHS_DIR      = os.path.join(ROOT, "graphs")

# SBERT-RAG outputs
SBERT_RAG_CSV     = result("sbert_rag_results.csv")
SBERT_RAG_METRICS = result("sbert_rag_metrics.txt")

# Evaluation outputs
COMPARISON_CSV  = result("all_models_comparison.csv")
COMPARISON_TXT  = result("all_models_comparison.txt")
ERROR_REPORT    = result("error_analysis_report.txt")
