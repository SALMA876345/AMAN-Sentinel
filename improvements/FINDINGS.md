# Findings

Results of the improvement analyses. Every number here comes from a file in
`results/improvements/` and can be regenerated with
`improvements/run_improvements.ps1`.

---

## 1. The headline: the framework works in exactly one setting, and it is the one that was cut

`gate_vs_confidence.csv`. Twenty-four gate x setting comparisons, each reporting
the AURC advantage over a MaxProb confidence baseline with a 95% cluster
bootstrap over evidence groups.

**In 22 of 24 comparisons no gate beats plain softmax confidence: 16 are reliably
worse and 6 cannot be told apart from the baseline.** Both shipped gates lose in every in-domain setting:
FAAV-Light by -0.042 to -0.071 AURC, FAAV-Full by -0.027 to -0.031. A verifier
already knows which of its own answers are shaky; these gates throw that
information away and replace it with something worse.

The exception:

| Study | Task | Gate | AURC advantage | 95% CI | Wins in resamples |
|---|---|---|---|---|---|
| Zero-shot UAE | three-class | FAAV-Light | **+0.0689** | [+0.0512, +0.0937] | 400/400 |
| Zero-shot UAE | three-class | FAAV-Rich (trees) | **+0.0194** | [+0.0092, +0.0270] | 400/400 |

Under zero-shot domain shift the evidence gate ranks the verifier's errors
substantially better than the verifier's own confidence. The mechanism is the
expected one: softmax confidence is badly calibrated off-distribution, and an
evidence-side signal that never saw the training distribution degrades more
gracefully.

The report dropped this study. Section VI-A says it kept "the two experiments
that best support the assignment narrative" - and the discarded one is the only
place the proposed method beats a proper baseline.

**Recommendation: reinstate zero-shot as Study 1 and reframe the claim.**
Faithfulness gating earns its place under domain shift, not in-domain. That is a
narrower claim than the report currently implies and a much more defensible one.

---

## 2. Why the gate collapsed: a fixable bug, not a dead end

`verifier_replication_summary.json`. `final_studies_faav_binary.py` (lines
154-161) fits the gate's "is this verdict trustworthy?" target on the verifier's
predictions over the rows the verifier was fine-tuned on.

| Family | Task | Verifier accuracy where the gate fits | On held-out validation |
|---|---|---|---|
| source | binary | 0.9796 | 0.8961 |
| source | three-class | 0.9609 | 0.8735 |
| combined | binary | 0.9650 | 0.8605 |
| combined | three-class | 0.9563 | 0.8709 |

The gate is shown 2.0-4.4% negatives where reality is 10.4-14.0%, and those few
are memorisation residue rather than the errors the verifier makes on unseen
data. It learns "always accept," which is exactly what the Study 3 three-class
sweep selected.

Refitting out-of-fold on held-out rows and swapping the TF-IDF features - which
re-read the same claim and evidence text the verifier already consumes - for
confidence geometry, evidence structure and provenance recovers most of the gap
to the confidence baseline (FAAV-Rich, above).

---

## 3. Ties silently resolved in the proposed system's favour

`best_results_tie_aware.csv`. Three of six settings have a three-way tie at the
top in which the FAAV variants reproduce their own verifier row for row:

| Study | Task | Best macro F1 | Models at that score |
|---|---|---|---|
| Zero-shot UAE | binary | 0.5637 | AraBERT, FAAV-Full, FAAV-Light |
| Source-domain | binary | 0.9016 | AraBERT, FAAV-Full, FAAV-Light |
| Combined adaptation | three-class | 0.8763 | AraBERT, FAAV-Full, FAAV-Light |

`final_best_results.csv` names FAAV-Full the winner of the third, and report
Table 12 inherits it.

---

## 4. A provenance shortcut explains more of the label than the majority baseline admits

`control_baselines.csv`. A lookup table over the publisher, reading neither claim
nor evidence:

| Setting | Majority | Provenance only | Reported best | Headroom over provenance |
|---|---|---|---|---|
| Combined, three-class | 0.2033 | **0.6009** | 0.8763 | 0.2754 |
| Combined, binary | 0.3424 | 0.5026 | 0.8669 | 0.3643 |
| Source, three-class | 0.2287 | 0.3316 | 0.7610 | 0.4294 |
| Source, binary | 0.3460 | 0.5094 | 0.9016 | 0.3922 |

This strengthens the paper rather than weakening it. The models add 0.275 macro
F1 over a provenance shortcut in the headline setting - a real number, and one
that pre-empts the obvious objection that the labels are confounded with corpus
origin.

---

## 5. The test sets are far smaller than their row counts suggest

`evidence_structure.csv`. Evidence passages repeat, so rows are not independent
draws. Kish effective sample size over evidence clusters:

| Split | Rows | Effective N | CI inflation |
|---|---|---|---|
| uae_three_class_full | 6,390 | **26** | 15.7x |
| uae_binary_full | 5,053 | **27** | 13.6x |
| combined_three_class_test | 6,369 | 319 | 4.5x |
| combined_binary_test | 6,034 | 281 | 4.6x |
| source_three_class_test | 5,091 | 797 | 2.5x |
| source_binary_test | 5,023 | 800 | 2.5x |

One passage is reused 1,009 times.

**The measured penalty is smaller than the Kish figure implies.** Kish effective
N is dominated by a handful of very large clusters, so it is a worst-case
structural warning rather than a prediction. Resampling evidence groups directly
(`bootstrap_intervals.csv`) gives the number to actually report:

| Setting | Macro F1 | Cluster 95% CI | Row CI width | Cluster CI width | Ratio |
|---|---|---|---|---|---|
| Zero-shot, three-class | 0.1412 | [0.1199, 0.1655] | 0.0165 | 0.0456 | **2.77x** |
| Zero-shot, binary | 0.5657 | [0.5351, 0.6005] | 0.0270 | 0.0654 | **2.42x** |
| Combined, three-class | 0.8754 | [0.8525, 0.8946] | 0.0174 | 0.0420 | **2.42x** |
| Combined, binary | 0.8659 | [0.8477, 0.8830] | 0.0173 | 0.0354 | **2.05x** |
| Source, three-class | 0.7778 | [0.7295, 0.8170] | 0.0652 | 0.0876 | 1.34x |
| Source, binary | 0.9012 | [0.8910, 0.9112] | 0.0165 | 0.0202 | 1.22x |

So intervals are 1.2-2.8 times wider than a naive row bootstrap, not the 2.5-16x
the Kish ratios suggest. Report the bootstrap numbers; cite effective N only as
a structural property of the corpus.

The practical consequence still bites. The combined three-class interval is
[0.8525, 0.8946], a span of 0.042 - wider than most of the between-model gaps in
Table 13, where AraBERT (0.8763), MARBERT (0.8577) and BiLSTM (0.8146) are
treated as an ordering. AraBERT and MARBERT are not separable at this
resolution, which is why the zero-shot finding in section 1 is reported with a
cluster bootstrap rather than a row bootstrap.

---

## 6. The entailment faithfulness metric was built, and it cannot yet be validated

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


`faithfulness_nli.csv`. The intended fix for the circular faithfulness column was
a multilingual NLI model scoring evidence against claim. It was run on all six
settings. **It fails its own validation and must not be reported as a
faithfulness measure.**

Against the 1,278 rows with silver relevance labels:

| Silver relevance | n | Neutral | Entailment |
|---|---|---|---|
| Irrelevant | 32 | 62.5% | 18.8% |
| Partially Relevant | 874 | 61.7% | 11.3% |
| Relevant | 372 | 53.5% | 8.6% |

Neutral share falls only 9 points across the whole relevance range, and
entailment share moves the wrong way. Zero-shot `mDeBERTa-v3-base-xnli` does not
transfer to Arabic fact-checking text of this kind.

The criticism of the shipped metric still stands - for gated rows it equals
coverage, so Table 15's faithfulness column is not comparable across rows. But
the replacement needs an Arabic NLI or stance model fine-tuned on the adjudicated
annotation subset and validated against human labels. Until then the honest move
is to report acceptance rate and the retrieval heuristic under their own names
and drop the word "faithfulness".

The failed attempt belongs in the limitations: it tells the next person that
off-the-shelf multilingual entailment is not a shortcut here.

---

## 7. Replication under a newer library stack

`verifier_replication_summary.json`. Retraining AraBERT with the identical
protocol under transformers 5.5 rather than 4.35:

| Setting | Replicated | Reported | Difference |
|---|---|---|---|
| Source-domain, binary | 0.9012 | 0.9016 | -0.0004 |
| Combined, binary | 0.8659 | 0.8609 | +0.0050 |
| Zero-shot, binary | 0.5657 | 0.5637 | +0.0020 |
| Zero-shot, three-class | 0.1412 | 0.1544 | -0.0132 |
| Source-domain, three-class | 0.7778 | 0.7437 | +0.0341 |

Binary reproduces within 0.005. Three-class moves by up to 0.034 - larger than
several differences the report treats as model effects. This is the concrete
argument for the multiple seeds and intervals the report says are needed.
`requirements-pinned.txt` records the exact stack.

---

## 8. Still outstanding

* **Inter-annotator agreement.** The instrument is built - 240 stratified rows,
  two blind sheets, key, codebook, in `annotation/`. Two people have to annotate
  before `score_annotations.py` will emit a kappa. It refuses to score blank
  sheets rather than produce a number that would be fabricated.
* **Collection time period.** Required by the brief; not derivable from the data.
* **Word count.** Body prose is 5,862 against a 6,000 floor. The new sections
  cover the shortfall.
