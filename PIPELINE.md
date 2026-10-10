# FAAV - Faithfulness-Aware Arabic Verification Pipeline

**Course:** AI503 Machine Learning  
**Assignment:** 2  
**Final framing:** research-first, true-test-first

## Project Goal

This project studies Arabic misinformation verification under domain shift. The
main contribution is **FAAV**, a faithfulness-aware retrieval-augmented
verification framework that adds evidence relevance, grounding, faithfulness,
and hallucination diagnostics to ordinary claim classification.

The final paper should **not** claim that the in-distribution 2-class result is
the final test result. Those high scores are kept as an internal diagnostic /
upper bound. The publishable result is the harder external-test story:

> Arabic verification performs well when train and test are distributionally
> similar, but drops sharply on independent UAE-domain testing. FAAV is valuable
> because it exposes evidence and faithfulness failure modes, not because it
> simply wins every raw-accuracy table.

## Implemented End-to-End Pipeline

The project now implements the complete faithfulness-controlled flow:

```text
Arabic Claim
    |
    v
Claim Cleaning and Normalisation
    |
    v
Evidence Retrieval (BM25, dense, or hybrid)
    |
    v
MARBERT Claim-Evidence Verification
    |
    v
Faithfulness and Evidence-Quality Checking
    |
    v
Final Decision Gate
    |
    +--> accept MARBERT Real/Fake verdict
    |
    +--> abstain to Unverified
```

The final decision accepts a `Real` or `Fake` verifier output only when all of
the following checks pass:

```text
Evidence length              >= 40 characters
MARBERT verifier confidence  >= 0.50
Claim-evidence similarity    >= 0.20
Explanation grounding        >= 0.30
```

Otherwise the final verdict becomes `Unverified`. Thresholds can be changed
without editing code:

```powershell
$env:MIN_EVIDENCE_CHARS="40"
$env:NLI_CONF_THRESH="0.50"
$env:EVIDENCE_SIM_THRESH="0.20"
$env:FAITH_THRESH="0.30"
python pipeline/faav_pipeline_full.py
```

The result CSV preserves both layers:

```text
verifier_prediction     Raw MARBERT/NLI verdict
verifier_confidence     Maximum verifier probability
faav_prediction         Final faithfulness-controlled verdict
final_decision_reason   accepted or the failed checks
decision_overridden     Whether FAAV replaced the verifier verdict
```

Implementation:

- `utils/faav_common.py`: `faithfulness_controlled_decision`
- `pipeline/faav_pipeline_full.py`: applies the gate to every claim
- `utils/test_final_decision.py`: unit tests for acceptance and abstention

**Result-version warning:** existing `results/faav_full_results*.csv` files were
generated before this final decision gate was added. Rerun
`pipeline/faav_pipeline_full.py` before reporting the new abstention rate,
override rate, or post-gate classification metrics.

## Folder Structure

```text
Code files/
  data/          Dataset files and constructed train/test splits
  results/       Model predictions, metrics, audits, and comparison tables
  baselines/     SVM, logistic regression, BiLSTM, AraBERT, MARBERT
  pipeline/      RAG, SBERT-RAG, FAAV-Light, FAAV-Full scripts
  eval/          Unified evaluation and error analysis
  utils/         Dataset construction, audits, and preprocessing utilities
  graphs/        Generated report figures
  config.py      Central path and experiment configuration
  generate_report.py
                 Builds the final Word report
```

## Final Evaluation Hierarchy

The final paper uses three studies. Each study is evaluated as a binary
claim-veracity task and a three-class faithfulness-aware decision task.

| Study | Training | Validation | Final test | Purpose |
|---|---|---|---|---|
| 1. Zero-shot UAE | Source train only | Source validation | Complete UAE task dataset | Full-domain cross-domain transfer |
| 2. Source-domain | Source train only | Source validation | Frozen source test | In-domain learnability |
| 3. Combined-domain | Source train + UAE train | Source + UAE validation | Source + UAE test | Full combined-data evaluation |

The source pool contains VERA-ARAB, AraStance, and ArCOV19. AFND is excluded
from verdict supervision. Study 1 evaluates all eligible UAE rows without UAE
training or threshold tuning: 5,053 binary rows and 6,390 three-class rows.
Study 3 uses grouped UAE train/validation/test partitions and evaluates on 20%
of the combined task dataset. Study 1 and Study 3 therefore answer different
questions and are not row-for-row comparable.

Primary files:

```text
results/final_studies/final_model_comparison.csv
results/final_studies/final_best_results.csv
results/final_studies/final_results_audit.json
results/final_studies/manifest_audit.json
```

The table in `results/zero_shot/` remains a legacy experiment. The primary
full-domain zero-shot results are rebuilt under `results/final_studies/` using
the same final model implementations and evaluation schema as Studies 2 and 3.

## Frozen Dataset Protocol

### Task definitions

**Binary task**

- Labels: `Real`, `Fake`
- UAE labels come from claim truth.
- Rows labelled Unverified are excluded.
- Binary FAAV always preserves a `Real` or `Fake` verdict and separately marks
  it as `Verified` or `Unverified`.
- FAAV binary evaluation therefore reports ordinary classification metrics
  for all rows plus coverage and selective metrics for `Verified` rows.

**Three-class task**

- Labels: `Real`, `Fake`, `Unverified`
- UAE mapping:
  - `Supported` -> `Real`
  - `Refuted` -> `Fake`
  - `Not Enough Evidence` -> `Unverified`
- This is the main FAAV task because the final decision depends on whether the
  available evidence supports committing to a verdict.

### Frozen task-specific source manifests

```text
Binary:
data/final_studies/source_binary_train.csv        17,580 rows (70.00%)
data/final_studies/source_binary_validation.csv    2,512 rows (10.00%)
data/final_studies/source_binary_test.csv          5,023 rows (20.00%)

Three-class:
data/final_studies/source_three_class_train.csv        17,817 rows (70.00%)
data/final_studies/source_three_class_validation.csv    2,545 rows (10.00%)
data/final_studies/source_three_class_test.csv          5,091 rows (20.00%)
```

- Split ratio: `70/10/20` for each task independently.
- Claims are grouped so one normalized claim cannot cross partitions.
- Seventy-two conflicting claim groups, representing 196 rows, are removed.
- Exact claim-group overlap across source partitions: `0`.

### Frozen task-specific UAE manifests

```text
Study 1 full-domain evaluation:
data/final_studies/uae_binary_full.csv             5,053 rows (100.00%)
data/final_studies/uae_three_class_full.csv        6,390 rows (100.00%)

Study 3 grouped partitions:
Binary:
data/final_studies/uae_binary_train.csv        3,537 rows (70.00%)
data/final_studies/uae_binary_validation.csv     505 rows (9.99%)
data/final_studies/uae_binary_test.csv          1,011 rows (20.01%)

Three-class:
data/final_studies/uae_three_class_train.csv        4,473 rows (70.00%)
data/final_studies/uae_three_class_validation.csv     639 rows (10.00%)
data/final_studies/uae_three_class_test.csv          1,278 rows (20.00%)
```

- Split ratio: `70/10/20` for each task independently.
- Study 1 uses the complete UAE task manifest; the 70/10/20 partitions are
  used only for Study 3.
- Repeated evidence passages are grouped into one partition.
- Evidence-group overlap across UAE partitions: `0`.
- Exact source-to-UAE claim overlap: `0`.
- The binary test contains 527 Real and 484 Fake rows.
- The three-class test contains 174 Real, 136 Fake, and 968 Unverified rows.

### Frozen combined-domain manifests

```text
Binary:
data/final_studies/combined_binary_train.csv        21,117 rows (70.00%)
data/final_studies/combined_binary_validation.csv    3,017 rows (10.00%)
data/final_studies/combined_binary_test.csv          6,034 rows (20.00%)

Three-class:
data/final_studies/combined_three_class_train.csv        22,290 rows (70.00%)
data/final_studies/combined_three_class_validation.csv    3,184 rows (10.00%)
data/final_studies/combined_three_class_test.csv          6,369 rows (20.00%)
```

Each combined partition is the union of its frozen source and UAE partition.
Protected claim/evidence-group overlap across combined partitions is `0`.

### Label limitation

VERA-ARAB and ArCOV19 provide claim-veracity labels. AraStance provides the
native evidence-stance and NEI-style supervision. Consequently, the source
three-class split contains only 68 Unverified test rows. UAE three-class labels
are evidence-conditioned silver labels derived from claim truth and
evidence-quality fields. This limitation must be stated in the paper.

## Final Results

### Best result in each setting

| Study | Task | Best model | N | Accuracy | Macro F1 | MCC |
|---|---|---|---:|---:|---:|---:|
| Zero-shot UAE | Binary | AraBERT | 5,053 | 0.5870 | 0.5637 | 0.1719 |
| Zero-shot UAE | Three-class | BiLSTM | 6,390 | 0.1592 | 0.1722 | 0.0400 |
| Source-domain | Binary | AraBERT | 5,023 | 0.9017 | 0.9016 | 0.8047 |
| Source-domain | Three-class | MARBERT | 5,091 | 0.9071 | 0.7610 | 0.8181 |
| Combined-domain | Binary | MARBERT | 6,034 | 0.8669 | 0.8669 | 0.7346 |
| Combined-domain | Three-class | FAAV-Full / AraBERT | 6,369 | 0.8789 | 0.8763 | 0.8092 |

### Main proposed system

The strongest combined-domain three-class result is shared by AraBERT,
FAAV-Light, and FAAV-Full:

```text
Accuracy          0.8789
Macro F1          0.8763
MCC               0.8092
Real F1           0.8737
Fake F1           0.8894
Unverified F1     0.8659
Unverified rate   0.1920
```

The validation-selected FAAV gates preserve the adapted AraBERT decisions, so
FAAV-Light and FAAV-Full tie the verifier. This is a neutral ablation result:
the gate does not reduce performance, but it also does not add a measurable
classification improvement on the combined test.

### Binary FAAV results

Binary FAAV uses AraBERT as the verifier and adds a separate faithfulness
status. Overall Accuracy, Macro F1, and MCC equal AraBERT by design because no
Real/Fake verdict is discarded. Coverage is the fraction marked `Verified`.

| Study | Gate | Test N | Coverage | Selective accuracy | Selective Macro F1 | Selective MCC |
|---|---|---:|---:|---:|---:|---:|
| Zero-shot UAE | FAAV-Light | 5,053 | 0.5587 | 0.6273 | 0.5641 | 0.2209 |
| Zero-shot UAE | FAAV-Full | 5,053 | 0.5575 | 0.6269 | 0.5640 | 0.2206 |
| Source-domain | FAAV-Light | 5,023 | 0.8256 | 0.9503 | 0.9503 | 0.9015 |
| Source-domain | FAAV-Full | 5,023 | 0.7961 | 0.9577 | 0.9577 | 0.9158 |
| Combined-domain | FAAV-Light | 6,034 | 0.7937 | 0.9140 | 0.9139 | 0.8294 |
| Combined-domain | FAAV-Full | 6,034 | 0.8011 | 0.9233 | 0.9232 | 0.8471 |

Thresholds are selected on validation only, with a minimum validation coverage
of `0.80`. The source-tuned gate falls to roughly `0.56` coverage on UAE,
providing additional evidence of cross-domain shift.

### Required evaluation metrics

Classification metrics are reported for every model:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Per-class Precision, Recall, and F1-score
- MCC and Weighted F1 as additional imbalance-aware measures

Verification metrics are reported for FAAV only on rows with UAE silver
annotations. Study 2 has no gold evidence annotations, so its verification
metrics are `N/A`. Study 3 verification values use only its frozen UAE subset,
not all source-plus-UAE rows.

| Study | Task | FAAV | Verification N | Evidence relevance | Faithfulness score | Hallucination rate | Citation support |
|---|---|---|---:|---:|---:|---:|---:|
| Zero-shot UAE | Binary | Light | 5,053 | 0.6465 | 0.4843 | 0.8572 | 0.4761 |
| Zero-shot UAE | Binary | Full | 5,053 | 0.6465 | 0.4847 | 0.8576 | 0.4769 |
| Zero-shot UAE | Three-class | Light | 6,390 | 0.6538 | 0.0987 | 0.9076 | 0.1524 |
| Zero-shot UAE | Three-class | Full | 6,390 | 0.6538 | 0.1027 | 0.9074 | 0.1564 |
| Combined-domain | Binary | Light | 1,011 | 0.6657 | 0.6142 | 0.8214 | 0.6497 |
| Combined-domain | Binary | Full | 1,011 | 0.6657 | 0.5984 | 0.8295 | 0.5254 |
| Combined-domain | Three-class | Light | 1,278 | 0.6330 | 0.8678 | 0.6893 | 0.7841 |
| Combined-domain | Three-class | Full | 1,278 | 0.6330 | 0.8678 | 0.6893 | 0.7841 |

Definitions:

- Evidence relevance: mean quality weight, with Relevant=`1`, Partially
  Relevant=`0.5`, and Irrelevant=`0`.
- Faithfulness score: accuracy of FAAV's High/Low status against the silver
  faithfulness label.
- Hallucination rate: Low-faithfulness evidence among committed Real/Fake
  outputs.
- Citation support accuracy: agreement between the final output and
  Supported/Refuted/Not Enough Evidence on rows containing an evidence URL.

The hallucination rate is an operational unsupported-commitment measure. It
does not claim that FAAV generated fabricated prose, because the final-study
system predicts verdicts rather than long-form answers.

### Error analysis

The best model from each of the six final settings is analyzed on its
misclassified rows. Categories are multi-label heuristic indicators and should
not be presented as human gold causal labels.

| Category | Tagged error instances | Share of 10,001 error instances |
|---|---:|---:|
| Retrieval failure | 6,068 | 60.7% |
| Political ambiguity | 4,970 | 49.7% |
| Conflicting evidence | 1,923 | 19.2% |
| Dialect failure | 889 | 8.9% |
| Temporal misinformation | 200 | 2.0% |
| Sarcasm | 39 | 0.4% |

Because an error may receive several tags, percentages do not sum to 100%.
Zero-shot UAE is especially retrieval-heavy: retrieval indicators occur in
`62.8%` of binary errors and `67.9%` of three-class errors. Combined adaptation
reduces the three-class retrieval-error share to `30.4%`.

Generated files:

```text
results/final_studies/final_error_analysis_by_setting.csv
results/final_studies/final_error_category_summary.csv
results/final_studies/final_error_analysis_rows.csv
results/final_studies/final_error_examples.csv
results/final_studies/final_error_analysis_definitions.json
graphs/final_studies/04_error_category_rates.png
```

### Main findings

- The three classes are learnable in-domain: source-domain MARBERT reaches
  `0.9071` accuracy and `0.7610` macro F1.
- Strict full-domain zero-shot three-class transfer fails: the best macro F1 is
  only `0.1722`.
- Study 3 reaches `0.8789` accuracy and `0.8763` macro F1 for three classes.
- MARBERT gives the strongest combined binary result at `0.8669` macro F1.
- Combined binary FAAV-Full reaches `0.9232` selective macro F1 at `0.8011`
  coverage while retaining AraBERT's unconditional binary verdicts.
- MARBERT remains the strongest source-domain three-class model.
- Study 3's high result is a mixed source+UAE evaluation and must not be
  described as UAE-only external performance.

### Legacy tables

Files outside `results/final_studies/` are retained for historical comparison.
They may use different sample sizes, labels, or protocols and must not be mixed
into the primary ranking.

## Reproducible Run Order

### Full final workflow

```powershell
.\run_final_studies.ps1
```

This command:

1. Builds and audits the frozen source and UAE manifests.
2. Tunes and runs LR and Linear SVM.
3. Trains BiLSTM, AraBERT, and MARBERT with validation early stopping.
4. Evaluates FAAV-Light and FAAV-Full.
5. Computes verification metrics and error analysis.
6. Consolidates all metrics and figures.
7. Runs the final-decision unit tests.

The complete run is compute-intensive. Completed components can be retained:

```powershell
.\run_final_studies.ps1 -SkipClassical -SkipNeural -SkipFAAV
```

This rebuilds the manifests, comparison tables, figures, and tests using
existing model outputs.

### Individual commands

```powershell
python utils/build_final_study_manifests.py
python experiments/final_studies_classical.py
python experiments/final_studies_neural.py
python experiments/final_studies_faav.py
python experiments/final_studies_faav_binary.py
python utils/build_final_study_results.py
python utils/evaluate_final_verification_metrics.py
python utils/analyze_final_errors.py
python generate_final_report.py
```

The neural runner supports resume-friendly subsets:

```powershell
python experiments/final_studies_neural.py --models bilstm
python experiments/final_studies_neural.py --models arabert --tasks three_class --families combined
```

### Final outputs

```text
results/final_studies/final_model_comparison.csv
results/final_studies/final_model_comparison.md
results/final_studies/final_best_results.csv
results/final_studies/final_results_audit.json
results/final_studies/manifest_audit.json
results/final_studies/predictions/
results/final_studies/models/
```

Figures:

```text
graphs/final_studies/01_best_macro_f1_by_study.png
graphs/final_studies/02_combined_domain_macro_f1.png
graphs/final_studies/03_three_class_accuracy_vs_macro_f1.png
graphs/final_studies/04_error_category_rates.png
```

Final Word report:

```text
Assignment2_Final_Report_Full_UAE.docx
```

## Recommended Report Structure

1. Introduction and research problem
2. Related Arabic misinformation and verification work
3. FAAV architecture
4. Dataset roles and label harmonisation
5. Frozen leakage-control protocol
6. Study 1: zero-shot cross-domain evaluation
7. Study 2: source-domain evaluation
8. Study 3: combined-domain evaluation
9. Binary versus three-class analysis
10. FAAV gate ablation
11. Error analysis and class imbalance
12. Limitations and ethical considerations
13. Conclusion

The abstract and conclusion should lead with Study 3, while Study 1 supplies the
domain-shift motivation and Study 2 demonstrates that the task is learnable.

## Remaining Publication Work

The computational study is complete. The remaining improvements concern data
quality and scientific validation:

- Obtain human labels for UAE evidence relevance and faithfulness.
- Increase native source-domain Unverified/NEI supervision.
- Improve diverse UAE Fake-claim evidence coverage and independent annotation.
- Replace repeated snippets with article-level evidence and gold evidence IDs.
- Report confidence intervals or repeated-seed variance.
- Add qualitative error examples for false Real, false Fake, and excessive
  abstention.
- Compare against an independently implemented Arabic NLI or fact-verification
  baseline if publication scope requires it.

## Final Paper Position

The defensible research claim is:

> FAAV is a faithfulness-aware framework for Arabic misinformation verification
> that supports Real, Fake, and Unverified decisions. Frozen experiments show
> strong source-domain learnability, severe zero-shot degradation in the UAE
> domain, and substantial recovery through combined source-plus-UAE adaptation.
> The zero-shot study evaluates all 5,053 binary and 6,390 three-class UAE
> rows, while the source and combined studies use grouped 70/10/20 splits. The
> combined-domain three-class system reaches 87.89% accuracy and 87.63% macro
> F1. Strict UAE-only zero-shot performance remains weak, so the high combined
> result should be interpreted as mixed-domain performance rather than external
> UAE generalisation.
