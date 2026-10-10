"""Content for Paper B: selective prediction for Arabic claim verification.

Same style-neutral form as Paper A, and it reuses Paper A's reference entries
where the two overlap, adding the selective-prediction and calibration work
this paper needs. Paper B cites Paper A for the dataset rather than repeating
its audits.
"""
from __future__ import annotations

from improvements.paper_a_content import REFERENCES as SHARED

OUT_STEM = "PaperB_Selective"

TITLE = ("When Does Evidence Gating Help? A Selective-Prediction Evaluation "
         "of Faithfulness Gates for Arabic Claim Verification")

AUTHORS = [
    ("Hamza Alqaruoti", 1),
    ("Mohammed Alqaryouti", 1),
    ("Manar Alkhatib", 2),
]
AFFILIATIONS = [
    "Faculty of Computer Science and AI, The British University in Dubai",
    "Department of Computer Science and AI, The British University in Dubai",
]

ABSTRACT = (
    "A misinformation system that must sometimes decline to answer needs a way "
    "to decide when. Two are available: the verifier's own confidence, which "
    "is free, and a faithfulness gate that inspects the retrieved evidence, "
    "which is the design most retrieval-grounded frameworks adopt. The second "
    "is rarely measured against the first. We evaluate three faithfulness "
    "gates for Arabic claim verification as selective predictors, using "
    "risk-coverage curves and area under them rather than full-test macro F1, "
    "which cannot reward abstention and drives threshold selection toward "
    "accepting everything. Across 24 gate and setting comparisons on AFVD-Core, "
    "a gate improves on softmax confidence in 2; in 16 it is reliably worse and "
    "in 6 indistinguishable. The two wins are both under zero-shot transfer, "
    "where a rule-based gate improves the area under the risk-coverage curve "
    "by 0.0689 (95% CI 0.0512 to 0.0937 over a bootstrap of evidence groups). "
    "We trace the in-domain failure to a concrete cause: the learned gate fits "
    "its trustworthiness target on the rows the verifier was fine-tuned on, "
    "where the verifier is 96 to 98% accurate, so it sees 2 to 4% errors where "
    "the true rate is 10 to 14% and learns to accept everything. Refitting on "
    "held-out data with features the verifier cannot already see recovers most "
    "of the gap but never overtakes confidence in domain. The practical rule "
    "we draw is narrow and testable: apply evidence-side gating where a "
    "verifier operates away from its training distribution and its confidence "
    "has stopped ranking its errors, and use confidence thresholding "
    "otherwise."
)

KEYWORDS = ("selective prediction; abstention; calibration; Arabic NLP; fact "
            "verification; retrieval-grounded verification; risk-coverage")

# ---------------------------------------------------------------- references
REFERENCES = dict(SHARED)
REFERENCES.update({
    "afvd2026": {
        "inline": "Alqaruoti et al., 2026",
        "ieee": 'H. Alqaruoti, M. Alqaryouti, and M. Alkhatib, "AFVD-Core: An '
                'Arabic claim-evidence verification dataset and an audit of what '
                'its labels measure," 2026, companion paper.',
        "acl": "Hamza Alqaruoti, Mohammed Alqaryouti, and Manar Alkhatib. 2026. "
               "AFVD-Core: An Arabic claim-evidence verification dataset and an "
               "audit of what its labels measure. Companion paper.",
    },
    "elyaniv2010": {
        "inline": "El-Yaniv and Wiener, 2010",
        "ieee": 'R. El-Yaniv and Y. Wiener, "On the foundations of noise-free '
                'selective classification," J. Mach. Learn. Res., vol. 11, '
                'pp. 1605-1641, 2010.',
        "acl": "Ran El-Yaniv and Yair Wiener. 2010. On the foundations of "
               "noise-free selective classification. Journal of Machine Learning "
               "Research, 11:1605-1641.",
    },
    "geifman2017": {
        "inline": "Geifman and El-Yaniv, 2017",
        "ieee": 'Y. Geifman and R. El-Yaniv, "Selective classification for deep '
                'neural networks," in Adv. Neural Inf. Process. Syst., 2017.',
        "acl": "Yonatan Geifman and Ran El-Yaniv. 2017. Selective classification "
               "for deep neural networks. In Advances in Neural Information "
               "Processing Systems.",
    },
    "geifman2019": {
        "inline": "Geifman et al., 2019",
        "ieee": 'Y. Geifman, G. Uziel, and R. El-Yaniv, "Bias-reduced uncertainty '
                'estimation for deep neural classifiers," in Proc. Int. Conf. '
                'Learn. Representations, 2019.',
        "acl": "Yonatan Geifman, Guy Uziel, and Ran El-Yaniv. 2019. Bias-reduced "
               "uncertainty estimation for deep neural classifiers. In "
               "Proceedings of ICLR.",
    },
    "hendrycks2017": {
        "inline": "Hendrycks and Gimpel, 2017",
        "ieee": 'D. Hendrycks and K. Gimpel, "A baseline for detecting '
                'misclassified and out-of-distribution examples in neural '
                'networks," in Proc. Int. Conf. Learn. Representations, 2017.',
        "acl": "Dan Hendrycks and Kevin Gimpel. 2017. A baseline for detecting "
               "misclassified and out-of-distribution examples in neural "
               "networks. In Proceedings of ICLR.",
    },
    "guo2017": {
        "inline": "Guo et al., 2017",
        "ieee": 'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration '
                'of modern neural networks," in Proc. Int. Conf. Mach. Learn., '
                '2017.',
        "acl": "Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. "
               "On calibration of modern neural networks. In Proceedings of ICML.",
    },
    "ovadia2019": {
        "inline": "Ovadia et al., 2019",
        "ieee": "Y. Ovadia et al., \"Can you trust your model's uncertainty? "
                'Evaluating predictive uncertainty under dataset shift," in Adv. '
                'Neural Inf. Process. Syst., 2019.',
        "acl": "Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, "
               "Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, and "
               "Jasper Snoek. 2019. Can you trust your model's uncertainty? "
               "Evaluating predictive uncertainty under dataset shift. In "
               "Advances in Neural Information Processing Systems.",
    },
    "angelopoulos2023": {
        "inline": "Angelopoulos and Bates, 2023",
        "ieee": 'A. N. Angelopoulos and S. Bates, "Conformal prediction: A gentle '
                'introduction," Found. Trends Mach. Learn., 2023.',
        "acl": "Anastasios N. Angelopoulos and Stephen Bates. 2023. Conformal "
               "prediction: A gentle introduction. Foundations and Trends in "
               "Machine Learning.",
    },
    "lewis2020": {
        "inline": "Lewis et al., 2020",
        "ieee": 'P. Lewis et al., "Retrieval-augmented generation for '
                'knowledge-intensive NLP tasks," in Adv. Neural Inf. Process. '
                'Syst., 2020.',
        "acl": "Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, "
               "Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, "
               "Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. "
               "2020. Retrieval-augmented generation for knowledge-intensive NLP "
               "tasks. In Advances in Neural Information Processing Systems.",
    },
    "robertson2009": {
        "inline": "Robertson and Zaragoza, 2009",
        "ieee": 'S. Robertson and H. Zaragoza, "The probabilistic relevance '
                'framework: BM25 and beyond," Found. Trends Inf. Retr., 2009.',
        "acl": "Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic "
               "relevance framework: BM25 and beyond. Foundations and Trends in "
               "Information Retrieval.",
    },
    "reimers2019": {
        "inline": "Reimers and Gurevych, 2019",
        "ieee": 'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings '
                'using Siamese BERT-networks," in Proc. EMNLP, 2019.',
        "acl": "Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence "
               "embeddings using Siamese BERT-networks. In Proceedings of EMNLP.",
    },
    "abdulmageed2021": {
        "inline": "Abdul-Mageed et al., 2021",
        "ieee": 'M. Abdul-Mageed, A. Elmadany, and E. M. B. Nagoudi, "ARBERT and '
                'MARBERT: Deep bidirectional transformers for Arabic," in Proc. '
                'ACL, 2021.',
        "acl": "Muhammad Abdul-Mageed, AbdelRahim Elmadany, and El Moatez "
               "Billah Nagoudi. 2021. ARBERT and MARBERT: Deep bidirectional "
               "transformers for Arabic. In Proceedings of ACL.",
    },
    "ji2023": {
        "inline": "Ji et al., 2023",
        "ieee": 'Z. Ji et al., "Survey of hallucination in natural language '
                'generation," ACM Comput. Surveys, vol. 55, no. 12, 2023.',
        "acl": "Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, "
               "Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. 2023. "
               "Survey of hallucination in natural language generation. ACM "
               "Computing Surveys, 55(12).",
    },
})

# ---------------------------------------------------------------- body
BODY = [
    ("h1", "Introduction"),
    ("p",
     "A verification system deployed on real claims will meet evidence that "
     "settles nothing. Returning a confident verdict there is the failure mode "
     "that matters most, because a wrong label on truthful speech is costly "
     "and a reader has no way to see that the evidence was thin. The remedy is "
     "abstention, and abstention needs a rule for when to abstain."),
    ("p",
     "Two rules are available. The verifier already produces a probability "
     "distribution, and thresholding it is the standard baseline for detecting "
     "its own errors {c:hendrycks2017}. Alternatively a faithfulness gate can "
     "inspect the retrieved evidence and refuse to commit when the evidence "
     "looks unusable. Retrieval-grounded frameworks generally adopt the "
     "second, on the intuition that grounding is what makes a verdict "
     "trustworthy. The two are seldom compared, and the comparison is the "
     "point of this paper."),
    ("p",
     "We study three gates for Arabic claim verification on AFVD-Core "
     "{c:afvd2026}: a rule gate over retrieval confidence, claim-evidence "
     "overlap and evidence length; a learned gate over claim and evidence "
     "text; and a corrected learned gate introduced here. We evaluate them as "
     "selective predictors {c:elyaniv2010}{c:geifman2017} rather than as "
     "classifiers, because full-test macro F1 cannot reward abstention and, as "
     "we show, drives threshold selection toward accepting every row."),
    ("p",
     "Our findings are these. Across 24 gate and setting comparisons, a gate "
     "beats softmax confidence in 2, is reliably worse in 16, and is "
     "indistinguishable in 6. Both wins are under zero-shot transfer. The "
     "in-domain failure has a concrete and fixable cause in how the learned "
     "gate is trained, which we isolate. A corrected gate recovers most of the "
     "lost ground without overtaking confidence in domain. The resulting rule "
     "is narrow enough to be tested elsewhere: use evidence-side gating when "
     "the verifier is off its training distribution, and confidence otherwise."),

    ("h1", "Related Work"),
    ("p",
     "Selective classification formalises prediction with an abstention option "
     "and evaluates it by the trade-off between coverage and risk "
     "{c:elyaniv2010}{c:geifman2017}, summarised by the area under the "
     "risk-coverage curve {c:geifman2019}. The simplest selector is the "
     "model's own maximum softmax probability {c:hendrycks2017}, which remains "
     "hard to beat when the model is calibrated. Modern networks are typically "
     "overconfident {c:guo2017}, and their calibration degrades further under "
     "distribution shift {c:ovadia2019}, which is the mechanism we appeal to "
     "when explaining why the ordering between our selectors reverses. "
     "Conformal methods offer coverage guarantees under exchangeability "
     "{c:angelopoulos2023} and are complementary to what we measure here."),
    ("p",
     "Retrieval-grounded verification pairs a claim with retrieved passages "
     "{c:thorne2018}{c:lewis2020}, using lexical {c:robertson2009} or dense "
     "{c:reimers2019} retrieval. Arabic work builds on pretrained encoders "
     "such as AraBERT {c:antoun2020} and MARBERT {c:abdulmageed2021}. "
     "Faithfulness and hallucination have been studied mainly for generation "
     "{c:ji2023}, with measurement frameworks for attribution "
     "{c:rashkin2023} and factual consistency {c:honovich2022}. Our question "
     "is narrower: given a gate that claims to enforce grounding, does it "
     "order a verifier's errors better than the verifier's own confidence?"),

    ("h1", "Setup"),
    ("h2", "Data and settings"),
    ("p",
     "All experiments use AFVD-Core {c:afvd2026}, an Arabic claim-evidence "
     "dataset with grouped splits and published test fingerprints. Three "
     "settings are evaluated. The source-domain setting trains and tests "
     "within the source corpora. The combined setting adds a UAE target-domain "
     "partition to training and tests on the union. The zero-shot setting "
     "trains on the source corpora alone and tests on the complete UAE task "
     "dataset, with no UAE example seen in training. Each setting is run as a "
     "binary Real and Fake task and as a three-class task that adds "
     "Unverified."),
    ("p",
     "The companion paper reports an audit of the dataset's labels, which "
     "bears on how the absolute scores below should be read: more than half "
     "the rows labelled Real or Fake carry evidence that cannot settle them "
     "{c:afvd2026}. That does not affect the comparison in this paper, which "
     "is between selectors scored on identical rows, but it does mean the "
     "absolute macro F1 values should not be read as pure evidence-based "
     "verification."),

    ("h2", "Verifier and gates"),
    ("p",
     "The verifier is AraBERT {c:antoun2020} fine-tuned per setting, selected "
     "on validation macro F1. Three gates sit on top of it. FAAV-Light is a "
     "deterministic rule over verifier confidence, lexical overlap between "
     "claim and evidence, and evidence length, with thresholds tuned on "
     "validation. FAAV-Full learns a trustworthiness classifier over a TF-IDF "
     "representation of the verdict, claim and evidence. FAAV-Rich is the "
     "corrected variant introduced here: it is fitted on the held-out "
     "validation partition rather than on the verifier's own training rows, "
     "and its features are ones the verifier cannot already see, namely the "
     "geometry of its probability distribution, evidence length, how often the "
     "retrieved passage recurs in the corpus, lexical overlap and publisher."),
    ("p",
     "Recovering row-level probabilities required retraining the verifiers "
     "under the original protocol, since the published pipeline retained only "
     "aggregate metrics. Every selector reported below is scored on the same "
     "retrained predictions, so the comparison is internally consistent even "
     "where absolute values differ slightly from previously published runs."),

    ("h1", "Why Not Full-Test Macro F1"),
    ("p",
     "Scoring an abstaining system against forced-choice models on full-test "
     "macro F1 cannot answer the question. Every abstention costs coverage, "
     "and nothing in the metric credits the errors the abstention removed. "
     "Threshold selection under that objective therefore drifts toward "
     "accepting everything, which is exactly what happened in our combined "
     "three-class setting: validation selection chose an all-accept "
     "configuration, and the gated systems reproduced the verifier's "
     "predictions row for row, reporting identical scores to four decimal "
     "places. Read as a classification result, that looks like a tie. Read "
     "correctly, it means the gate did not act."),
    ("p",
     "We therefore evaluate with risk-coverage curves, the area under them "
     "(AURC, lower is better) {c:geifman2019}, and selective scores at matched "
     "coverage, which is the only way to compare two selectors without one "
     "buying its score with abstention. Three confidence baselines are "
     "included: maximum class probability {c:hendrycks2017}, the margin "
     "between the top two classes, and negative predictive entropy. On a "
     "binary task these three induce identical rankings, so only the "
     "three-class settings separate them."),

    ("h1", "Results"),
    ("table", "AURC advantage of each gate over the maximum-probability "
              "baseline, with 95% intervals from a bootstrap over evidence "
              "groups. Positive favours the gate.",
     [["Setting", "FAAV-Light", "FAAV-Full", "FAAV-Rich (trees)",
       "FAAV-Rich (linear)"],
      ["Zero-shot, binary", "-0.0510", "-0.0514", "-0.0117", "-0.0199"],
      ["Zero-shot, three-class", "+0.0689", "-0.0707", "+0.0194", "+0.0027"],
      ["Source, binary", "-0.0420", "-0.0271", "-0.0047", "-0.0009"],
      ["Source, three-class", "-0.0585", "-0.0276", "-0.0059", "-0.0003"],
      ["Combined, binary", "-0.0466", "-0.0298", "-0.0059", "-0.0118"],
      ["Combined, three-class", "-0.0711", "-0.0285", "-0.0006", "-0.0053"]],
     True),
    ("p",
     "Twenty-two of the 24 comparisons give the gate no advantage. In 16 the "
     "interval lies entirely below zero, so the gate is reliably worse than "
     "using the verifier's own confidence, and in 6 it cannot be separated "
     "from the baseline. The shipped rule gate is the weakest of the four in "
     "every in-domain setting, losing between 0.042 and 0.071 AURC."),
    ("figure", "Risk-coverage curves in the combined setting. A gate that "
               "helps sits below the MaxProb curve.",
     "risk_coverage.png", True),
    ("p",
     "The two exceptions are both under zero-shot transfer, on the three-class "
     "task. There FAAV-Light improves AURC by 0.0689, with an interval from "
     "0.0512 to 0.0937, and FAAV-Rich by 0.0194 (0.0092 to 0.0270). The "
     "direction reverses precisely where the verifier leaves its training "
     "distribution."),
    ("p",
     "Abstention itself is clearly worth having, whichever selector supplies "
     "it. At 80% coverage in the combined binary setting, thresholding the "
     "verifier's confidence lifts selective macro F1 from 0.8659 at full "
     "coverage to 0.9366, against 0.8974 for the shipped learned gate. The "
     "question this paper answers is not whether to abstain but what to "
     "abstain on."),
    ("figure", "Selective macro F1 at matched coverage in the combined "
               "setting.",
     "selective_gain.png", True),

    ("h1", "Why the Learned Gate Fails In Domain"),
    ("p",
     "The all-accept outcome is not a property of evidence gating. It follows "
     "from where the learned gate is fitted. FAAV-Full trains a classifier to "
     "predict whether the verifier's verdict is correct, and fits it on the "
     "rows the verifier was fine-tuned on. The verifier is nearly perfect "
     "there, so the target is almost constant."),
    ("table", "Verifier accuracy on the rows where the gate is fitted, against "
              "held-out accuracy.",
     [["Verifier", "Task", "Where the gate is fitted", "Held out",
       "Errors seen", "True error rate"],
      ["Source", "Binary", "0.9796", "0.8961", "2.0%", "10.4%"],
      ["Source", "Three-class", "0.9609", "0.8735", "3.9%", "12.7%"],
      ["Combined", "Binary", "0.9650", "0.8605", "3.5%", "14.0%"],
      ["Combined", "Three-class", "0.9563", "0.8709", "4.4%", "12.9%"]],
     True),
    ("p",
     "The gate sees 2 to 4% negative examples where the true rate on unseen "
     "data is 10 to 14%, and the few it does see are memorisation residue "
     "rather than the errors the verifier makes in deployment. It learns, "
     "correctly for its training distribution and uselessly for any other, to "
     "accept everything. Refitting on the held-out validation partition and "
     "replacing the TF-IDF view of the same text the verifier already consumes "
     "with signals it cannot see recovers most of the gap: FAAV-Rich comes "
     "within 0.0006 AURC of the confidence baseline in the combined "
     "three-class setting. It does not overtake it in any in-domain setting."),
    ("p",
     "The likely explanation for the zero-shot reversal is calibration. A "
     "fine-tuned transformer's probabilities stop tracking its errors once it "
     "leaves the distribution it was trained on {c:guo2017}{c:ovadia2019}, "
     "while evidence-side signals never depended on that distribution and "
     "degrade more gently. This is a hypothesis consistent with our numbers "
     "rather than a controlled result, and testing it directly, by measuring "
     "calibration error alongside AURC across a shift gradient, is the "
     "clearest next experiment."),

    ("h1", "Statistical Inference"),
    ("p",
     "Evidence passages repeat in AFVD-Core, one of them 1,009 times, so test "
     "rows are not independent and intervals from resampling rows are too "
     "narrow. All intervals in this paper come from a bootstrap over evidence "
     "groups {c:efron1993}, which widens 95% intervals on macro F1 by 1.22 to "
     "2.77 times relative to a row-level bootstrap. Kish effective sample size "
     "{c:kish1965} over those clusters is smaller still, and is reported in "
     "the companion paper {c:afvd2026}."),
    ("p",
     "Two consequences follow. In the combined three-class setting the "
     "interval on macro F1 spans 0.042, which exceeds most differences between "
     "the models we compare, so model orderings at that resolution should not "
     "be reported as findings. And for the gated systems against their own "
     "verifier in the all-accept settings, a paired test is undefined: the "
     "predictions agree on every row, so there is no difference to test. That "
     "is the rigorous way to state a result that classification metrics "
     "present as a tie."),
    ("p",
     "One further caution concerns reproducibility. Retraining the verifiers "
     "under a newer library version reproduced the binary results within 0.005 "
     "macro F1 but moved a three-class result by 0.034, which is larger than "
     "several differences commonly attributed to model choice. Single-seed "
     "three-class comparisons at that resolution are not stable."),

    ("h1", "Discussion"),
    ("p",
     "The result is narrower than the framing usually given to faithfulness "
     "gates, and more useful for it. Inside its training distribution a "
     "verifier already knows which of its answers are shaky, and a gate built "
     "over evidence heuristics discards that knowledge and replaces it with "
     "something weaker. Outside that distribution the verifier's confidence "
     "stops being informative, and the evidence signal, which never depended "
     "on the training distribution, becomes the better guide."),
    ("p",
     "Three recommendations follow for work on retrieval-grounded "
     "verification. Evaluate abstention with risk-coverage analysis rather "
     "than full-test classification metrics, which cannot reward it. Always "
     "report the softmax-confidence selector as a baseline, since a gate that "
     "does not beat it is not earning its place. And fit any learned gate on "
     "data the underlying model did not train on, because a trustworthiness "
     "target estimated on memorised rows is close to constant."),

    ("h1", "Limitations"),
    ("p",
     "The results come from one dataset and one language, with one verifier "
     "architecture and a single random seed. The zero-shot advantage rests on "
     "one target domain, so whether it generalises to other Arabic domains is "
     "untested and is the first thing we would check. We evaluate three gate "
     "designs; a better gate may exist, and our claim is about these designs "
     "and the way they are trained rather than about evidence gating in "
     "principle. The calibration explanation is consistent with the evidence "
     "but not directly measured. Finally, the absolute scores inherit the "
     "label limitations documented for the dataset {c:afvd2026}, which is why "
     "this paper's conclusions rest on comparisons between selectors scored on "
     "identical rows rather than on absolute values."),

    ("h1", "Conclusion"),
    ("p",
     "We evaluated faithfulness gates for Arabic claim verification as "
     "selective predictors against the verifier's own confidence. In 22 of 24 "
     "comparisons the gate gave no advantage, and in 16 it was reliably worse. "
     "The exception is zero-shot transfer, where a rule-based gate improves "
     "the risk-coverage area by 0.0689 with an interval excluding zero. We "
     "identified the cause of the in-domain failure in how the learned gate is "
     "fitted, and showed that correcting it recovers most but not all of the "
     "gap. Evidence-side gating belongs where a verifier's confidence has "
     "stopped ranking its errors, and confidence thresholding belongs "
     "everywhere else."),
]
