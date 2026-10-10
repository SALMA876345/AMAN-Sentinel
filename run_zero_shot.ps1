$ErrorActionPreference = "Stop"

python utils\build_zero_shot_assets.py --passage-max-chars 280 --passage-min-chars 40
python utils\tune_zero_shot_thresholds.py

$common = @(
    "experiments\faav_cross_domain.py",
    "--experiment", "cross_domain_uae",
    "--label-space", "verification",
    "--class-mode", "3",
    "--train-file", "data\verification_training_zeroshot.csv",
    "--test-file", "data\uae_zero_shot_test.xlsx",
    "--train-label-col", "verification_label",
    "--test-label-col", "verification_label",
    "--training-dedup", "pair",
    "--output-dir", "results\zero_shot"
)

python @common --model claim_only_lr --retrieval none
python @common --model claim_evidence_lr --retrieval none
python @common --model fusion_lr --retrieval none
python @common --model two_stage_lr --retrieval none `
    --gate-source learned --gate-model lr --gate-threshold 0.44 `
    --relevance-train-file data\relevance_training_zeroshot.csv

# Laptop retrieval profile. The complete corpus contains more than 46k passages;
# increase --corpus-limit on a faster server.
python @common --model two_stage_lr --retrieval bm25 `
    --retrieval-corpus-file data\retrieval_corpus_zeroshot.csv `
    --corpus-limit 1000 --top-k 3 `
    --gate-source learned --gate-model lr --gate-threshold 0.44 `
    --relevance-train-file data\relevance_training_zeroshot.csv

python @common --model two_stage_lr --retrieval tfidf `
    --retrieval-corpus-file data\retrieval_corpus_zeroshot.csv `
    --corpus-limit 46449 --top-k 3 `
    --gate-source learned --gate-model lr --gate-threshold 0.44 `
    --relevance-train-file data\relevance_training_zeroshot.csv

python eval\summarize_zero_shot.py
