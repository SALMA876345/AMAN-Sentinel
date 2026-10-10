# Report corrections

Specific edits to `FAAV_Assignment2_Hamza_Alqaruoti_Submission_Ready.docx`.
Paragraph numbers are the index into `docx.Document(...).paragraphs`.

**Two versions exist.** The 20-page paper at the top of `Assignment 2/`
(`A Faithfulness-Aware Retrieval-Grounded Framework for Arabic Misinformation
Verification.docx`, the one with "BM25 + FAAV-Light Gate" naming) carries every
problem below at different paragraph numbers:

| Problem | Submission_Ready | Top-level paper |
|---|---|---|
| Gate credited with the 978 Unverified rows | ¶147 | ¶161 |
| Hallucination 0.1054 -> 0.0311 "by abstaining on weak evidence" | ¶163 | ¶179 |
| Same claim repeated in the RQ2 answer | ¶171 | ¶187 |
| Faithfulness 0.8678 reported as a result | ¶167 | ¶184 |
| FAAV-Full selective reliability, no confidence baseline | ¶145 | ¶83, ¶159 |

Every claim in the submitted report was checked against the shipped metric
JSONs and reproduces exactly. These corrections are not about arithmetic. Three
of them fix statements whose *mechanism* is wrong, and the rest add controls
and inference the conclusions need in order to carry.

---

## 1. Paragraph 147 contradicts paragraph 146 - required fix

**Current text (¶147, final sentences):**

> The three-class FAAV-Full matrix shows much stronger abstention behaviour than
> the source-only setting: 978 of 1,036 Unverified rows are correctly retained
> as Unverified. This is the clearest empirical support for the FAAV idea,
> because the system is not only choosing between Real and Fake; it is learning
> when evidence is not sufficient for a trustworthy commitment.

**Why it is wrong.** In Study 3 three-class the validation sweep selected
`confidence_threshold = 0.0`, `overlap_threshold = 0.0`,
`evidence_char_threshold = 0` for FAAV-Light and `gate_threshold = 0.0` for
FAAV-Full. Both gates accept every row. The 978 correct Unverified predictions
are produced by the three-class AraBERT verifier, not by the faithfulness gate,
which abstains on nothing in this setting. The `abstention_rate` of 0.192
recorded in the metric JSON is the verifier's own rate of predicting
Unverified, not a gate abstention rate. ¶146 already says the gate "preserves
the adapted AraBERT decisions"; ¶147 then credits the gate with the result.

**Replacement text:**

> The three-class confusion matrix shows strong abstention behaviour: 978 of
> 1,036 Unverified rows are correctly retained as Unverified. This behaviour
> belongs to the adapted three-class verifier rather than to the faithfulness
> gate, which the validation sweep left in an all-accept configuration for this
> setting; the 0.192 abstention rate is the verifier's own rate of predicting
> Unverified. The result therefore supports the three-class formulation of the
> task - that a system given enough evidence-insufficiency examples learns when
> not to commit - while leaving the marginal contribution of the gate to the
> selective-prediction analysis in Section VII-E.

---

## 2. Paragraphs 163 and 171 misstate the hallucination mechanism - required fix

**Current text (¶163, final sentence):**

> In Study 2 three-class, the gate slightly improves macro F1 from 0.8193 to
> 0.8228 and reduces the operational hallucination proxy from 0.1054 to 0.0311
> by abstaining on weak evidence.

**Why it is wrong.** Two separate problems.

*The two numbers are not the same quantity.* In
`experiments/final_studies_rag.py` (lines 605-616) faithfulness is computed one
way for ungated systems and another for gated ones:

```python
if accepted is None:                     # Standard BM25 RAG, SBERT-RAG
    faithfulness_score = float(np.mean(predicted_supported))
else:                                    # FAAV-Light RAG
    faithfulness_score = float(np.mean(accepted))
"Hallucination Rate": float(1.0 - faithfulness_score)
```

For BM25, 0.1054 is the share of rows failing a retrieval-grounding heuristic.
For FAAV-Light RAG, 0.0311 is one minus its acceptance rate. Faithfulness for a
gated row is coverage restated, so hallucination is the abstention rate under
another name. The column is not comparable across the rows of Table 15.

*The stated mechanism runs backwards.* FAAV-Light RAG abstains on 3.11% of
rows; Standard BM25 RAG returns Unverified on 16.17%. The gated system abstains
**less**, so the improvement cannot be "by abstaining on weak evidence."

**Replacement text:**

> In Study 2 three-class the gate improves macro F1 from 0.8193 to 0.8228. The
> faithfulness and hallucination columns are not directly comparable between
> gated and ungated rows, because for a gated system they restate coverage
> rather than measuring grounding; Section VII-F reports an entailment-based
> faithfulness score computed identically for every system, which is the figure
> that should be compared across rows.

Apply the same correction to the RQ2 answer in ¶171, which repeats the
0.1054 -> 0.0311 claim.

**Table 15 fix.** Either split the column into `Faithfulness (grounding
heuristic)` and `Acceptance rate`, reporting each only where it is defined, or
replace both with the entailment score from
`results/improvements/faithfulness_nli.csv`, which is defined identically for
every system.

---

## 3. Ties resolved silently in favour of the proposed system - required fix

`results/final_studies/final_best_results.csv` names one winner per setting.
In three of the six settings the top score is a three-way tie in which the FAAV
variants reproduce their own verifier row for row:

| Study | Task | Best macro F1 | Models at that score |
|---|---|---|---|
| Study 1 - Zero-shot UAE | binary | 0.5637 | AraBERT, FAAV-Full, FAAV-Light |
| Study 2 - Source-domain | binary | 0.9016 | AraBERT, FAAV-Full, FAAV-Light |
| Study 3 - Combined adaptation | three-class | 0.8763 | AraBERT, FAAV-Full, FAAV-Light |

Table 12 inherits the choice and shows FAAV-Full as the Study 2 three-class
winner, so the proposed system appears to win a setting in which it did not
act. Use `results/improvements/best_results_tie_aware.csv`, which names every
tied model, and add a footnote to Table 12:

> Where the proposed system is listed, its score is identical to the AraBERT
> verifier it wraps; the gate was left in an all-accept configuration by
> validation selection and did not alter any prediction.

---

## 4. Binary FAAV is identical to AraBERT by construction - clarify

Binary FAAV returns the AraBERT verdict unchanged and adds a separate
Verified/Unverified status. This is deliberate and is documented in the module
docstring, but a reader of Table 13 sees three rows with identical numbers and
no explanation. Add one sentence beneath Table 13:

> By construction the binary FAAV variants return the AraBERT verdict and add a
> separate faithfulness status, so their unconditional classification scores are
> identical to AraBERT's by design; the variants differ only in coverage and
> selective metrics, reported in Table 13b.

---

## 5. New material to add

### 5.1 Provenance control baseline (Section VII-A)

`results/improvements/control_baselines.csv`. The report's only floor is the
majority class, which is weak: `Unverified` comes almost entirely from UAE
official and news feeds while Real/Fake come from VERA-ARAB, ArCOV19 and
AraStance, so publisher identity alone carries much of the label. A lookup
table over `source` that never reads the claim or the evidence reaches 0.6009
macro F1 on the Study 3 three-class test against the best model's 0.8763.

Reporting this strengthens rather than weakens the result: it shows the models
add 0.275 macro F1 over a provenance shortcut, which is the number a sceptical
reader wants, and it pre-empts the objection that the labels are confounded
with corpus origin. Figure: `figures/control_baselines.png`.

### 5.1b Root cause of the all-accept gate (Section VII-C or Limitations)

`results/improvements/verifier_replication_summary.json`. The FAAV-Full gate
fits its "is this verdict trustworthy?" target on `train_inferred` - the
verifier's predictions on the rows the verifier was fine-tuned on
(`experiments/final_studies_faav_binary.py`, lines 154-161). The verifier is
near-perfect there, so the target is almost constant and the gate learns to
accept everything:

| Family | Task | Verifier accuracy on train (where the gate fits) | On held-out validation |
|---|---|---|---|
| source | binary | 0.9796 | 0.8961 |
| source | three-class | 0.9609 | 0.8735 |
| combined | binary | 0.9650 | 0.8605 |
| combined | three-class | 0.9563 | 0.8709 |

The gate sees 2.0-4.4% negative examples where reality is 10.4-14.0%, and those
few negatives are memorisation residue rather than the errors the verifier makes
on new data. This is a concrete, fixable bug, not an inherent limit of the
approach: fitting the gate out-of-fold on held-out rows is what
`improvements/selective_evaluation.py` does, and it recovers most of the gap to
the confidence baseline.

### 5.2 Selective prediction (new Section VII-E)

`results/improvements/selective_aurc.csv`,
`selective_at_matched_coverage.csv`, `figures/risk_coverage.png`,
`figures/selective_gain.png`.

The submitted protocol scores an abstaining system against forced-choice models
on full-test macro F1. Under that objective a gate can only lose coverage
without being credited for the errors it removes, which is why the Study 3
sweep settles at the all-accept corner. Replacing it with risk-coverage
analysis, AURC, and selective scores at matched coverage measures the thing the
framework was built to do, and includes the confidence baselines - MaxProb,
margin, entropy - that a reviewer will ask for first, since a learned gate has
to beat plain softmax confidence to justify its existence.

### 5.3 Entailment faithfulness - attempted, and it cannot yet be validated

> **Revised after inspecting the annotation sample.** This section originally
> concluded that the NLI measure *fails* validation. That judgement used the
> silver relevance labels as the reference, and those labels are themselves
> unreliable: they grade a Dubai-airport drone claim paired with Epstein-leak
> evidence as "Relevant". A reference that calls unrelated evidence relevant
> cannot adjudicate a measure that calls it neutral. The two disagree and
> neither is trustworthy yet; the annotation study decides. Report neither as
> a faithfulness score. See `annotation/codebook.md` (v2) for why the labels
> are unreliable: 54.3% of Real/Fake test rows carry evidence that copies the
> claim or runs to five words or fewer.


`results/improvements/faithfulness_nli.csv`. The plan was to replace the
gate-dependent proxy with a multilingual NLI model reading evidence as premise
and claim as hypothesis, scored identically for every system. **It was run and
it fails its own validation check, so it must not be reported as a faithfulness
measure.**

Checked against the 1,278 rows carrying silver `retrieval_quality` labels, the
NLI relation barely tracks evidence relevance, and one component runs backwards:

| Silver relevance | n | Neutral | Entailment |
|---|---|---|---|
| Irrelevant | 32 | 62.5% | 18.8% |
| Partially Relevant | 874 | 61.7% | 11.3% |
| Relevant | 372 | 53.5% | 8.6% |

Neutral share falls only 9 points from Irrelevant to Relevant, and entailment
share *decreases* as relevance increases, which is the opposite of what a
working entailment measure does. Against the gold labels the separation is
similarly thin (Real 26.5% entailment vs Fake 16.8%). Zero-shot
`mDeBERTa-v3-base-xnli` is not a valid instrument on Arabic fact-checking text
of this kind.

**What this means for the report.** The criticism in item 2 stands unchanged -
the shipped faithfulness column is circular for gated rows and must not be
compared across Table 15. But the replacement is not available yet. Two honest
options:

* Report only the acceptance rate and the retrieval-grounding heuristic, each
  labelled for what it is, and drop the word "faithfulness" from the column
  header until there is a validated measure behind it.
* Build the measure properly: fine-tune an Arabic NLI or stance model on the
  adjudicated subset from the annotation study in 5.5, then validate it against
  those human labels before any faithfulness number is reported.

The failed attempt is itself worth a paragraph in the limitations. It documents
that off-the-shelf multilingual entailment does not transfer to this task,
which is a result the next person building an Arabic faithfulness metric needs.

### 5.3b The one place the framework demonstrably works

`results/improvements/gate_vs_confidence.csv`. Across 24 gate x setting
comparisons, the AURC advantage over the MaxProb confidence baseline is
reported with a 95% cluster-bootstrap interval over evidence groups.

**In 22 of 24 comparisons no gate beats plain softmax confidence: 16 are reliably
worse and 6 are indistinguishable from it.** Both shipped gates lose in every in-domain
setting - FAAV-Light by -0.042 to -0.071 AURC, FAAV-Full by -0.027 to -0.031.
The verifier already knows which of its answers are shaky, and the gate is
discarding that information.

The exception is the single setting the report deleted:

| Study | Task | Gate | AURC advantage | 95% CI |
|---|---|---|---|---|
| Zero-shot UAE | three-class | FAAV-Light | **+0.0689** | [+0.0512, +0.0937] |
| Zero-shot UAE | three-class | FAAV-Rich (trees) | **+0.0194** | [+0.0092, +0.0270] |

Under zero-shot domain shift the evidence gate ranks the verifier's errors
substantially better than the verifier's own confidence, and the advantage
survives resampling in 100% of 400 cluster bootstraps. The mechanism is the
expected one: a transformer's softmax confidence is badly calibrated off its
training distribution, so an evidence-side signal that never saw that
distribution degrades more gracefully.

**This is the paper's actual contribution, and it is in the study that was cut
for not fitting the narrative.** Section VI-A describes dropping the zero-shot
experiment to keep "the two experiments that best support the assignment
narrative"; the dropped experiment is the only one in which the proposed method
beats a proper baseline. Reinstating it as Study 1 and reframing the claim -
faithfulness gating earns its place under domain shift, not in-domain - turns a
null result into a defensible and more interesting one.

### 5.4 Statistical inference (Section VII-G and Limitations)

`results/improvements/significance_tests.csv`,
`bootstrap_intervals.csv`, `evidence_structure.csv`,
`figures/effective_sample.png`.

The report states that confidence intervals and significance tests are required
before small differences are read as real, then reports point estimates. Two
additions close that gap:

* **Paired McNemar tests.** All models in a setting share one test manifest, so
  the paired test is the correct one. FAAV against its own verifier in Study 3
  three-class is the degenerate case - zero disagreements in both cells, no test
  computable - which is the rigorous way to state the tie in ¶146.
* **Cluster-aware intervals.** Evidence passages repeat up to 1,009 times, so
  rows are not independent. Resampling evidence groups instead of rows widens
  the 95% interval on macro F1 by roughly 4.5x on the combined three-class test.
  The zero-shot UAE set is the extreme case: 6,390 rows resolve to about 26
  independent evidence clusters, so no conclusion drawn from it should be stated
  as a precise quantity.

### 5.5 Annotation and inter-annotator agreement (Section V, new subsection)

`results/improvements/annotation/`. The brief requires an Annotation subsection
covering labelling method, guidelines, and agreement; the report has none, and
the silver labels' quality is asserted in the limitations rather than measured.
The directory contains a stratified 240-row sample (40 per label x corpus
cell), two blind annotation sheets, a separate answer key, and a codebook.

Two people must fill the sheets before `score_annotations.py` will report
anything. That step cannot be automated away: a kappa computed without a second
annotator would be a fabricated measurement, which is worse than the current
omission.

---

## 6. Smaller items

* **Word count.** Body prose is 5,862 words against a 6,000-8,000 requirement;
  the total reaches 7,155 only by counting table cells. Sections VII-E and V's
  annotation subsection add roughly the shortfall.
* **Collection time period.** The brief asks for it explicitly under Data
  Collection. Section V-F names sources but no date range.
* **Table 5 examples.** Currently "constructed and anonymized". Real rows drawn
  from the corpus, anonymised, would evidence the dataset contribution that
  carries 20% of the mark.
* **Stale result files.** `final_model_comparison.md`, `final_best_results.csv`
  and `final_error_category_summary.csv` still use the three-study numbering
  the report abandoned. The error summary reports 10,001 error instances and
  60.7% retrieval failure; Table 17 reports 2,541 and 43.6%. Both are correct
  for their own scope, but a reader opening the folder sees a contradiction.
  Add a `SCOPE.md` to the results directory, or regenerate for the two retained
  studies.
* **Training-set sizes differ by model family.** Classical baselines train on
  25,474 rows (train + validation), neural models on 22,290 (train only), in the
  combined three-class setting. It favours the baselines, which still lose, so
  no conclusion changes - but ¶120's "fair comparison" should say so explicitly.
* **Runbook.** `run_final_studies.ps1` omits `experiments/final_studies_rag.py`,
  which produces Tables 14 and 15, and its final step calls
  `generate_final_report.py`, which is absent from the submission zip and writes
  a different filename than the report that was submitted. The actual generator
  is `generate_new_assignment.py`, which `SUBMISSION_README.md` lists as
  included but which is not in the archive.
* **Missing artifacts.** The zip ships no `predictions/` directory and no model
  checkpoints, so `final_studies_faav.py`, `final_studies_faav_binary.py`,
  `build_final_study_results.py`, `evaluate_final_verification_metrics.py` and
  `analyze_final_errors.py` all fail from a clean extract. The prediction dumps
  are small and would restore most of the pipeline.
