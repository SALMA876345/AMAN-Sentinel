"""
Integrity check: is the UAE-MVD test `evidence_text` safe to feed a verifier,
or does it LEAK the gold label (e.g. an LLM-written justification)?
Output -> results/evidence_integrity.txt
"""
import io, re
import pandas as pd

out = io.StringIO()
def P(*a): print(*a, file=out)

te = pd.read_excel("data/uae_test_dataset_full30k.xlsx", sheet_name="Tabelle1")
te = te[te["gold_label"].isin(["Real", "Fake"])].copy()
te["ev"] = [str(x) if x is not None else "" for x in te["evidence_text"].tolist()]
te["gold_label"] = [str(x) for x in te["gold_label"].tolist()]

P("rows (Real/Fake):", len(te))
P("\n--- label_source ---")
P(te["label_source"].value_counts(dropna=False).to_string())
P("\n--- evidence_url present by label ---")
te["has_url"] = [len(str(x).strip()) > 0 and str(x) != "nan" and str(x) != "None"
                 for x in te["evidence_url"].tolist()]
P(te.groupby("gold_label")["has_url"].mean().round(3).to_string())

# Verdict words that, if present, could reveal the answer
fake_terms = ["كاذب", "زائف", "مفبرك", "مضلل", "شائعة", "خاطئ", "عار عن الصحة",
              "false", "fake", "misleading", "لا أساس", "غير صحيح"]
true_terms = ["صحيح", "حقيقي", "مؤكد", "ثبت", "true", "verified", "أكد", "صحة الخبر"]
def has_any(s, terms): return any(t in s for t in terms)

te["ev_says_fake"] = te["ev"].apply(lambda s: has_any(s, fake_terms))
te["ev_says_true"] = te["ev"].apply(lambda s: has_any(s, true_terms))

P("\n--- LEAKAGE TEST: do verdict words in evidence track the gold label? ---")
ct = pd.crosstab(te["gold_label"], te["ev_says_fake"], normalize="index").round(3)
P("P(evidence contains a 'FALSE/fake' verdict word | label):")
P(ct.to_string())
ct2 = pd.crosstab(te["gold_label"], te["ev_says_true"], normalize="index").round(3)
P("\nP(evidence contains a 'TRUE/verified' verdict word | label):")
P(ct2.to_string())

# crude leak score: how often a naive 'verdict word' rule would guess the label
naive = te.apply(lambda r: "Fake" if r["ev_says_fake"] and not r["ev_says_true"]
                 else ("Real" if r["ev_says_true"] and not r["ev_says_fake"] else "?"), axis=1)
decided = naive != "?"
acc = (naive[decided] == te["gold_label"][decided]).mean()
P(f"\nNaive verdict-word rule decides {decided.mean()*100:.0f}% of claims, "
  f"and is {acc*100:.0f}% correct on those.")
P("  -> If this is very high, evidence_text leaks the label (verdict baked in).")
P("  -> If near 50% or low coverage, evidence is genuine external text (safe).")

# samples
P("\n--- SAMPLE evidence (first 200 chars) ---")
for lbl in ["Real", "Fake"]:
    P(f"\n[{lbl}]")
    for _, r in te[te["gold_label"] == lbl].head(3).iterrows():
        P("  CLAIM: " + str(r["claim_ar"])[:90])
        P("  EVID : " + r["ev"][:200].replace("\n", " "))
        P("  URL  : " + str(r["evidence_url"])[:80])
        P("")

with open("results/evidence_integrity.txt", "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("Saved -> results/evidence_integrity.txt")
