# FAAV improvements package

Additive analyses for the Assignment 2 submission. Everything here writes under
`results/improvements/`. No file in `results/final_studies/` or
`data/final_studies/` is modified, so the graded artifacts stay as they were.

## Why this exists

The submitted report's numbers are correct - every table and prose figure was
re-checked against the shipped metric JSONs and reproduces exactly, and the
split integrity claims (zero claim overlap, 20% test shares, 68 Unverified
source rows) verify independently. The problems are elsewhere:

1. **The proposed system does not act in half the settings.** Binary FAAV
   returns the AraBERT verdict by design. In Study 3 three-class the validation
   sweep also selected an all-accept gate, so FAAV-Light, FAAV-Full and AraBERT
   are identical to sixteen decimal places - yet FAAV-Full is named the winner.
2. **The evaluation cannot detect a working gate.** Scoring an abstaining system
   against forced-choice models on full-test macro F1 penalises every
   abstention and credits none, so the tuning sweep drifts to all-accept by
   construction.
3. **The gate is fed information it cannot use.** `FAAV-Full`'s TF-IDF gate
   reads the same claim and evidence text the AraBERT verifier already consumes,
   in a weaker representation, and fits its "is this verdict trustworthy?"
   target on the verifier's own training rows - where the verifier is near
   perfect, making the target almost constant.
4. **Faithfulness is defined circularly for gated systems**, so the RAG table's
   faithfulness and hallucination columns mean different things in different
   rows.
5. **No inference.** The report says confidence intervals and significance tests
   are needed, then reports point estimates - on test sets whose effective
   sample size is a fraction of their row count.

## Running it

```powershell
# Everything, including the ~20 minute verifier retrain.
powershell -File improvements/run_improvements.ps1

# Re-run the analyses only, once predictions exist.
powershell -File improvements/run_improvements.ps1 -SkipTraining
```

Requires a CUDA GPU for stage 1 and the NLI step. `aubmindlab/bert-base-arabertv02`
must be in the local HuggingFace cache; the NLI step downloads
`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (~210 MB) on first
run.

## What each module does

| Module | Question it answers |
|---|---|
| `train_verifiers.py` | Recovers the row-level probabilities the submission does not ship, using the exact protocol of `experiments/final_studies_neural.py`. Also records verifier accuracy on its own training rows, which is what makes the FAAV-Full gate target degenerate. |
| `control_baselines.py` | How much of each label is recoverable from the publisher alone, with no text? |
| `evidence_structure.py` | How many independent evidence clusters is each test set really made of? |
| `tie_aware_results.py` | Which "best model" selections are actually ties? |
| `selective_evaluation.py` | Does any gate rank its own errors better than plain softmax confidence? |
| `significance.py` | Are the reported differences distinguishable from noise once rows are treated as clustered? |
| `faithfulness_nli.py` | Is a committed verdict actually entailed by its evidence - measured the same way for every system? |
| `annotation_sample.py`, `score_annotations.py` | The inter-annotator agreement study the Dataset section is missing. |
| `make_figures.py` | Report-ready figures for the above. |

## Outputs

```
results/improvements/
  control_baselines.csv              majority / dataset-family / provenance controls
  evidence_structure.csv             effective N and CI inflation per split
  best_results_tie_aware.csv         best-model table with ties named
  selective_aurc.csv                 AURC and E-AURC per selector
  selective_at_matched_coverage.csv  selective scores at 70/80/90% coverage
  risk_coverage_curves.csv           curve data behind the figure
  significance_tests.csv             paired McNemar
  bootstrap_intervals.csv            cluster-aware vs row-level 95% intervals
  faithfulness_nli.csv               entailment faithfulness, gate-independent
  predictions/                       row-level probabilities per study and task
  figures/                           four PNGs
  annotation/                        blind sheets, key, codebook, agreement summary
```

## One thing this package will not do

`score_annotations.py` refuses to emit a kappa from blank sheets. Two people
have to annotate the 240-row sample first. An agreement statistic invented
without a second annotator would be a fabricated measurement, and the Dataset
section is better off with an honest gap than with one.

See `REPORT_CORRECTIONS.md` for the specific paragraph-level edits these results
imply.
