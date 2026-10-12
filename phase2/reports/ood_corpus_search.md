# Phase 2, Part B: controlled OOD corpus search

## What this is (and is not)

This report documents a search of the repository/dataset tree for a genuine,
on-disk, Arabic, **unused-in-training** corpus that could be used to sanity-check
the Part A novelty detector (`phase2/novelty_detector.py`) under real
distribution shift. It is **not** a labeled misinformation OOD benchmark, and
no such benchmark was fabricated. See the closing statement at the end.

## What was searched

1. **`Dataset files/AraNEWS/`** — checked `hackathon/restore_four_dataset_training.py`
   and `hackathon/restore_claim_training_fast.py` line-by-line for any reference to
   AraNEWS. Confirmed: neither file imports, opens, or mentions AraNEWS anywhere.
   `restore_four_dataset_training.py` only builds from `AFND`, `ArCOV19-Rumors`
   (`tweet_verification` + `claim_verification`), `VERA-ARAB`
   (`peerj-cs-10-2432-s001.xlsx` only), and `AraStance`
   (`arastance_{train,valid,test}.csv`). `restore_claim_training_fast.py` only
   consumes the already-unified table from those same four source families
   (`CLAIM_TRUTH_SOURCES = ["VERA-ARAB", "AraStance", "ArCOV19-claims",
   "ArCOV19-tweets"]` plus AFND `Unverified`). **AraNEWS is confirmed never
   referenced anywhere in the training-data-building code path.**
   AraNEWS's own task (per `Dataset files/AraNEWS/readme.md` and its WANLP 2020
   paper, "Machine Generation and Detection of Arabic Manipulated and Fake
   News") is **Original (human-written) vs. Machine (machine-generated) text**
   — a different task from Real/Fake misinformation verification, and its
   `label` column carries no misinformation ground truth at all.

2. **`Dataset files/VERA-ARAB/` supplementary files `s002`–`s008`** — confirmed
   only `peerj-cs-10-2432-s001.xlsx` is read by `build_vera_arab()` in
   `restore_four_dataset_training.py`. Inspected the rest directly:
   - `s002.xlsx` — a 2-column geo codebook (`location`, `iso_a3`); not a text corpus.
   - `s003.txt` — a plain-text list of required Python packages (setup notes).
   - `s004.zip` — three Excel **codebooks** (`Columns Heading`, `locations-iso-mapping`,
     `VERA-ARAB_dataset`), i.e. metadata/documentation for s001, not new text.
   - `s005.py`–`s008.py` — Python analysis/plotting scripts from the original
     VERA-ARAB paper authors (stats, maps, wordclouds), not data files.
   None of `s002`–`s008` contains a usable, distinctly different Arabic text
   corpus — they are all either non-text metadata or code.

3. No other unused, on-disk, Arabic text corpus was found elsewhere in the
   repository tree beyond what is already consumed by the restoration scripts.

## What was selected and why

**`Dataset files/AraNEWS/ArNews_test_GW_order_0/ArNews_test_GW_order_0.tsv`**
(despite the `.tsv` extension, the file is comma-delimited with quoted fields;
columns: `Unnamed: 0`, `SentID`, `Sentence`, `label`). This is the best fit
available: it is real, on-disk, Arabic, confirmed unused anywhere in building
`data/claim_training.csv` or the clean split, and its `Sentence` text is
qualitatively different from the clean split's social-media/news-headline-style
claims (longer, more formal newswire sentences).

Only a **bounded prefix** of the file was read (`nrows=5000`, not the full
~4.9 MB file), from which a reproducible random sample of **500 rows** was
drawn with `random_state=42`. The sample's `label` field (Orginal/Machine) was
inspected only to describe the corpus — it was never used in scoring or in any
threshold decision.

## Method

1. Loaded the Part A fitted detector via
   `phase2.novelty_detector.load_fitted_detector()` — this reuses the exact
   persisted TRAIN reference embeddings (`artifacts/novelty/train_reference_embeddings.npy`)
   and the exact percentile thresholds from `artifacts/novelty/config.json`
   (P80 = 0.275643, P95 = 0.371312 novelty_score). No refitting, no new
   thresholds.
2. Embedded the 500-row AraNEWS `Sentence` sample with the same SBERT model/call
   signature and scored it with the same k=10 mean-cosine-distance method.
3. Compared the resulting FAMILIAR / MODERATELY NOVEL / HIGHLY NOVEL category
   shares against the clean-split **TEST** scores already persisted in
   `artifacts/novelty/test_scores.csv` (in-distribution reference point).

## Comparison table

| Category          | Clean TEST (n=4000) | Clean TEST % | AraNEWS sample (n=500) | AraNEWS % |
|--------------------|---------------------|--------------|--------------------------|-----------|
| FAMILIAR           | 3231                | 80.77%       | 85                       | 17.0%     |
| MODERATELY NOVEL   | 591                 | 14.77%       | 137                      | 27.4%     |
| HIGHLY NOVEL       | 178                 | 4.45%        | 278                      | 55.6%     |

Mean novelty_score: clean TEST = 0.1697 (median 0.1626) vs. AraNEWS sample =
0.3841 (median 0.3899) — roughly double, and the AraNEWS sample's median alone
sits above the HIGHLY NOVEL cutoff (0.3713).

## Interpretation

The novelty detector's score distribution shifts sharply and in the expected
direction: claims drawn from the same source domains as training (clean TEST)
are overwhelmingly scored FAMILIAR (~81%), while sentences from a corpus the
detector's reference set never saw (AraNEWS newswire text) are overwhelmingly
scored MODERATELY/HIGHLY NOVEL (~83%, with the majority HIGHLY NOVEL). This is
the expected qualitative behavior for a working novelty/OOD signal under real
text-distribution shift.

## Required closing statement

External cross-domain OOD classifier performance (e.g., on UAE-style data) is
**NOT YET ESTABLISHED** — no genuine labeled OOD misinformation benchmark
exists in this repository. The AraNEWS sample check above validates only that
the novelty detector's scores shift sensibly under text-distribution shift,
not that it would correctly gate misinformation-classifier errors on a
genuinely different domain. AraNEWS carries no Real/Fake misinformation
labels (its `label` column is Original-vs-Machine-generated text, an unrelated
task), so no "OOD classification accuracy" number can be or was computed from
it.
