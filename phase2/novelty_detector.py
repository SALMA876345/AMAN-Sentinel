"""
AMAN Sentinel - Phase 2, Part A: claim novelty / out-of-distribution (OOD) detector.

IMPORTANT FRAMING
-----------------
This module answers ONE question: "how unfamiliar is this input CLAIM relative to the
training distribution of claims the downstream Real/Fake classifier was fit on?"

It is a novelty / distribution-shift detector, NOT a misinformation detector.
    - novelty_score is NOT a fake-news probability.
    - A HIGHLY NOVEL claim is simply unlike anything seen in training (it could be
      true or false). A FAMILIAR claim is simply well-covered by the training
      distribution (also regardless of truth).
    - Downstream, novelty is meant to be used for ROUTING (e.g. "flag for human
      review because the Real/Fake model has little basis to judge this"), never
      as a verdict on veracity.

METHOD (kept deliberately simple, per the brief)
-------------------------------------------------
1. Text representation: embeds `claim_norm` ONLY (not `evidence_norm`).
   Reasoning: the question we care about is novelty of the INPUT CLAIM itself --
   i.e. "has the downstream classifier seen claims like this one?". The
   `evidence_norm` field in this restored dataset is retrieval-backfilled for a
   large share of rows (see hackathon/restore_claim_training_fast.py), so a claim's
   evidence text mostly reflects what the retrieval corpus happened to return, not
   a property of the claim. Folding evidence into the novelty representation would
   therefore partly measure "is this evidence passage covered by the retrieval
   corpus" rather than genuine claim novelty, and would make the score depend on
   retrieval-pipeline artifacts instead of the input the user actually submits.

2. Embedding model: reuses this repo's existing SBERT usage
   (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), encoded with the
   same call signature already used in pipeline/hybrid_retrieval.py
   (batch_size=64, show_progress_bar=False, normalize_embeddings=True).

3. Reference distribution: fit ONLY on TRAIN split claim_norm embeddings. No
   validation/test embeddings are ever added to the reference set.

4. Novelty score: mean COSINE DISTANCE to the k=10 nearest neighbors in the TRAIN
   reference embedding set (cosine distance = 1 - cosine_similarity; since all
   embeddings are L2-normalized by `normalize_embeddings=True`, this is a
   monotonic transform of Euclidean distance, so "cosine" vs "Euclidean on
   normalized vectors" is not a meaningful fork here -- cosine is used because it
   is the standard metric for SBERT similarity and what sklearn's NearestNeighbors
   exposes directly).
   This is the SIMPLEST defensible density/novelty estimator: a point deep inside
   a dense training region has small mean kNN distance (familiar); a point far
   from any training example has large mean kNN distance (novel). No Mahalanobis
   distance or Isolation Forest is implemented -- the brief explicitly asks not to
   add methods for complexity's sake, and kNN-distance is sufficient as a first pass.

5. Category thresholds: computed from PERCENTILES of the VALIDATION split's
   UNSUPERVISED novelty_score distribution only -- task_label is never consulted.
      score < P80(validation scores)                -> FAMILIAR
      P80(validation scores) <= score < P95(...)     -> MODERATELY NOVEL
      score >= P95(validation scores)                -> HIGHLY NOVEL
   The exact P80 / P95 values are computed at fit time and persisted in
   artifacts/novelty/config.json (see `thresholds`).

Persisted artifacts (artifacts/novelty/):
    - train_reference_embeddings.npy  : (n_train, dim) float32 L2-normalized SBERT
                                         embeddings of hackathon clean-split TRAIN
                                         claim_norm text. This IS the fitted
                                         reference distribution.
    - config.json                     : model name, representation choice +
                                         reasoning, k, distance metric, thresholds
                                         + derivation, random seed.
    - validation_scores.csv           : sample_id, novelty_score, novelty_category
                                         for hackathon/data/clean_split/
                                         source_binary_validation.csv
    - test_scores.csv                 : same, for source_binary_test.csv

Usage:
    python phase2/novelty_detector.py          # fit on train, score val/test, persist
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required for the novelty detector "
        "(same dependency already used by pipeline/hybrid_retrieval.py)"
    ) from exc

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_SPLIT_DIR = REPO_ROOT / "hackathon" / "data" / "clean_split"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "novelty"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_COLUMN = "claim_norm"
K_NEIGHBORS = 10
DISTANCE_METRIC = "cosine"
RANDOM_SEED = 42

CATEGORY_FAMILIAR = "FAMILIAR"
CATEGORY_MODERATE = "MODERATELY NOVEL"
CATEGORY_HIGH = "HIGHLY NOVEL"

# Percentile cutoffs applied to the VALIDATION split's unsupervised novelty_score
# distribution. Chosen so that, by construction on the validation set, ~80% of
# rows are FAMILIAR, ~15% MODERATELY NOVEL, ~5% HIGHLY NOVEL -- a reasonable
# "most inputs look like training, a shrinking tail looks progressively odder"
# shape for a first-pass routing signal. No labels are used to pick these numbers.
LOWER_PERCENTILE = 80.0
UPPER_PERCENTILE = 95.0


def _set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------


@dataclass
class NoveltyThresholds:
    lower_percentile: float
    upper_percentile: float
    lower_value: float
    upper_value: float

    def categorize(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        categories = np.full(scores.shape, CATEGORY_FAMILIAR, dtype=object)
        categories[scores >= self.lower_value] = CATEGORY_MODERATE
        categories[scores >= self.upper_value] = CATEGORY_HIGH
        return categories

    def to_dict(self) -> dict:
        return {
            "lower_percentile": self.lower_percentile,
            "upper_percentile": self.upper_percentile,
            "lower_value": self.lower_value,
            "upper_value": self.upper_value,
            "derivation": (
                "Computed from percentiles of the UNSUPERVISED novelty_score "
                "distribution over the VALIDATION split only (task_label was "
                "never consulted). score < lower_value -> FAMILIAR; "
                "lower_value <= score < upper_value -> MODERATELY NOVEL; "
                "score >= upper_value -> HIGHLY NOVEL."
            ),
        }


@dataclass
class NoveltyDetector:
    """kNN-distance novelty/OOD detector fit on TRAIN claim embeddings only.

    novelty_score(x) = mean cosine distance from x's embedding to its k nearest
    neighbors in the TRAIN reference embedding set. Higher = less familiar.

    This is NOT a misinformation/fake-news classifier. It never sees task_label.
    """

    model_name: str = MODEL_NAME
    k: int = K_NEIGHBORS
    metric: str = DISTANCE_METRIC
    seed: int = RANDOM_SEED

    _model: SentenceTransformer | None = field(default=None, repr=False, compare=False)
    _nn_index: NearestNeighbors | None = field(default=None, repr=False, compare=False)
    reference_embeddings: np.ndarray | None = field(default=None, repr=False, compare=False)
    thresholds: NoveltyThresholds | None = field(default=None, compare=False)

    # -- embedding -----------------------------------------------------
    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using this project's existing SBERT usage pattern
        (see pipeline/hybrid_retrieval.py, HybridRetriever dense branch)."""
        model = self._get_model()
        texts = [str(t) if t is not None else "" for t in texts]
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    # -- fitting ---------------------------------------------------------
    def fit_reference(self, reference_texts: list[str]) -> np.ndarray:
        """Fit the reference distribution from TRAIN texts only. Returns the
        embeddings so the caller can persist them."""
        _set_seed(self.seed)
        embeddings = self.embed(reference_texts)
        self.reference_embeddings = embeddings
        self._nn_index = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self._nn_index.fit(embeddings)
        return embeddings

    def load_reference(self, embeddings: np.ndarray) -> None:
        """Attach an already-fitted reference embedding matrix (e.g. loaded
        from artifacts/novelty/train_reference_embeddings.npy) without
        re-embedding or re-downloading the SBERT model's training text."""
        self.reference_embeddings = np.asarray(embeddings, dtype=np.float32)
        self._nn_index = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self._nn_index.fit(self.reference_embeddings)

    # -- scoring -----------------------------------------------------------
    def score_embeddings(self, query_embeddings: np.ndarray) -> np.ndarray:
        if self._nn_index is None:
            raise RuntimeError("Detector has no fitted reference. Call fit_reference() or load_reference() first.")
        distances, _ = self._nn_index.kneighbors(query_embeddings, n_neighbors=self.k)
        return distances.mean(axis=1)

    def score_texts(self, texts: list[str]) -> np.ndarray:
        return self.score_embeddings(self.embed(texts))

    # -- thresholds ----------------------------------------------------
    def fit_thresholds(self, validation_scores: np.ndarray) -> NoveltyThresholds:
        """Derive category cutoffs from PERCENTILES of an unsupervised score
        distribution (intended: validation novelty_score values). No labels
        of any kind are used here."""
        lower_value = float(np.percentile(validation_scores, LOWER_PERCENTILE))
        upper_value = float(np.percentile(validation_scores, UPPER_PERCENTILE))
        self.thresholds = NoveltyThresholds(
            lower_percentile=LOWER_PERCENTILE,
            upper_percentile=UPPER_PERCENTILE,
            lower_value=lower_value,
            upper_value=upper_value,
        )
        return self.thresholds

    def categorize(self, scores: np.ndarray) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("Thresholds not set. Call fit_thresholds() first.")
        return self.thresholds.categorize(scores)

    # -- persistence -------------------------------------------------------
    def to_config_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "text_representation": {
                "column": TEXT_COLUMN,
                "reasoning": (
                    "Embeds claim_norm only, not evidence_norm. Novelty of the "
                    "INPUT CLAIM is what matters for routing decisions. The "
                    "evidence_norm field in this restored dataset is "
                    "retrieval-backfilled for many rows (see "
                    "hackathon/restore_claim_training_fast.py), so including it "
                    "would partly measure retrieval-corpus coverage rather than "
                    "genuine claim novelty."
                ),
            },
            "k": self.k,
            "distance_metric": self.metric,
            "distance_metric_note": (
                "mean cosine distance (1 - cosine_similarity) to the k nearest "
                "neighbors in the TRAIN reference embedding set. Embeddings are "
                "L2-normalized by SentenceTransformer.encode(normalize_embeddings"
                "=True), so this is a monotonic transform of Euclidean distance "
                "on normalized vectors; cosine was used directly since it is "
                "sklearn.neighbors.NearestNeighbors' native metric name for it."
            ),
            "reference_split": "hackathon/data/clean_split/source_binary_train.csv (claim_norm column, all rows, TRAIN ONLY)",
            "random_seed": self.seed,
            "thresholds": self.thresholds.to_dict() if self.thresholds else None,
            "category_labels": [CATEGORY_FAMILIAR, CATEGORY_MODERATE, CATEGORY_HIGH],
            "warning": (
                "novelty_score / novelty_category are OOD/unfamiliarity signals, "
                "NOT misinformation scores. HIGHLY NOVEL does not mean 'Fake'; "
                "FAMILIAR does not mean 'Real'. task_label was never used to fit "
                "the reference distribution or the thresholds."
            ),
        }


# ---------------------------------------------------------------------------
# Part A pipeline: fit on train, score val/test, persist artifacts
# ---------------------------------------------------------------------------


def _load_clean_split(split: str) -> pd.DataFrame:
    path = CLEAN_SPLIT_DIR / f"source_binary_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected clean split file not found: {path}")
    return pd.read_csv(path)


def build_and_persist(artifacts_dir: Path = ARTIFACTS_DIR) -> NoveltyDetector:
    _set_seed(RANDOM_SEED)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clean split CSVs ...")
    train_df = _load_clean_split("train")
    val_df = _load_clean_split("validation")
    test_df = _load_clean_split("test")

    detector = NoveltyDetector()

    print(f"Embedding TRAIN claim_norm ({len(train_df)} rows) with {MODEL_NAME} ...")
    train_embeddings = detector.fit_reference(train_df[TEXT_COLUMN].tolist())
    np.save(artifacts_dir / "train_reference_embeddings.npy", train_embeddings)
    print(f"  saved {artifacts_dir / 'train_reference_embeddings.npy'} shape={train_embeddings.shape}")

    print(f"Embedding + scoring VALIDATION claim_norm ({len(val_df)} rows) ...")
    val_scores = detector.score_texts(val_df[TEXT_COLUMN].tolist())

    print("Deriving category thresholds from VALIDATION novelty_score percentiles (unsupervised, no labels) ...")
    thresholds = detector.fit_thresholds(val_scores)
    print(f"  P{LOWER_PERCENTILE:.0f}={thresholds.lower_value:.6f}  P{UPPER_PERCENTILE:.0f}={thresholds.upper_value:.6f}")

    val_categories = detector.categorize(val_scores)
    val_out = pd.DataFrame(
        {
            "sample_id": val_df["sample_id"].values,
            "novelty_score": val_scores,
            "novelty_category": val_categories,
        }
    )
    val_out.to_csv(artifacts_dir / "validation_scores.csv", index=False)
    print(f"  saved {artifacts_dir / 'validation_scores.csv'}")
    print("  validation category counts:", val_out["novelty_category"].value_counts().to_dict())

    print(f"Embedding + scoring TEST claim_norm ({len(test_df)} rows) ...")
    test_scores = detector.score_texts(test_df[TEXT_COLUMN].tolist())
    test_categories = detector.categorize(test_scores)
    test_out = pd.DataFrame(
        {
            "sample_id": test_df["sample_id"].values,
            "novelty_score": test_scores,
            "novelty_category": test_categories,
        }
    )
    test_out.to_csv(artifacts_dir / "test_scores.csv", index=False)
    print(f"  saved {artifacts_dir / 'test_scores.csv'}")
    print("  test category counts:", test_out["novelty_category"].value_counts().to_dict())

    config = detector.to_config_dict()
    with open(artifacts_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  saved {artifacts_dir / 'config.json'}")

    return detector


# ---------------------------------------------------------------------------
# Reload helper for other Phase-2 code (e.g. the Part B OOD corpus check)
# ---------------------------------------------------------------------------


def load_fitted_detector(artifacts_dir: Path = ARTIFACTS_DIR) -> NoveltyDetector:
    """Rebuild a NoveltyDetector from persisted artifacts (reference
    embeddings + config, including thresholds) WITHOUT re-embedding the
    training claims. Use this to score new/external text against the exact
    same fitted reference and thresholds produced by build_and_persist()."""
    config_path = artifacts_dir / "config.json"
    ref_path = artifacts_dir / "train_reference_embeddings.npy"
    if not config_path.exists() or not ref_path.exists():
        raise FileNotFoundError(
            f"Novelty artifacts not found under {artifacts_dir}. Run "
            "`python phase2/novelty_detector.py` first."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    detector = NoveltyDetector(
        model_name=config["model_name"],
        k=config["k"],
        metric=config["distance_metric"],
        seed=config["random_seed"],
    )
    reference_embeddings = np.load(ref_path)
    detector.load_reference(reference_embeddings)

    thr = config["thresholds"]
    detector.thresholds = NoveltyThresholds(
        lower_percentile=thr["lower_percentile"],
        upper_percentile=thr["upper_percentile"],
        lower_value=thr["lower_value"],
        upper_value=thr["upper_value"],
    )
    return detector


if __name__ == "__main__":
    build_and_persist()
