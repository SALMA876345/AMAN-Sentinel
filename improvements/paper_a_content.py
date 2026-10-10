"""Content for Paper A: the AFVD-Core resource paper and its label audit.

Written once, in a style-neutral form, so the same text can be typeset as an
IEEE conference paper or in ACL/LREC style. Citations appear as {c:key} for a
parenthetical reference and {t:key} for a textual one; the builder turns them
into "[7]" or "(Alhindi et al., 2021)" depending on the target.

Every figure in the text comes from results/improvements and can be
regenerated with improvements/run_improvements.ps1.
"""
from __future__ import annotations

OUT_STEM = "PaperA_AFVD"

TITLE = ("AFVD-Core: An Arabic Claim-Evidence Verification Dataset "
         "and an Audit of What Its Labels Measure")

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
    "Arabic misinformation resources are mostly claim-only: a post is labelled "
    "true or false, and no evidence is attached to justify the label. Systems "
    "trained on them cannot cite anything, and cannot be evaluated on whether "
    "their verdicts follow from evidence. We present AFVD-Core, an Arabic "
    "claim-evidence dataset that harmonises VERA-ARAB, AraStance and ArCOV19 "
    "with a UAE target-domain collection, attaching retrieved evidence, "
    "provenance, URLs and silver verification fields under grouped splits with "
    "published test fingerprints. We then audit what its labels actually "
    "measure, because a claim-evidence dataset assembled this way inherits "
    "claim-truth labels while presenting itself as evidence-based. Three "
    "audits are reported. A mechanical check finds that 54.3% of the rows "
    "labelled Real or Fake carry evidence that copies the claim or runs to "
    "five words or fewer, and so cannot settle it. A provenance control that "
    "reads no claim or evidence text reaches 0.6009 macro F1 against a 0.8763 "
    "model, so publisher identity alone recovers most of the distance a "
    "majority baseline suggests is being learned. A human audit of 240 "
    "stratified rows agrees with the silver verdicts at or below chance "
    "(kappa 0.037 and -0.110 across two passes), and judges 61 of 71 rows "
    "graded Relevant to be irrelevant. We also show that repeated evidence "
    "passages shrink the effective sample size of the test sets by up to a "
    "factor of 15, widening realistic confidence intervals by 1.2 to 2.8 "
    "times. The dataset is usable, and we release it with these measurements "
    "rather than without them, because they determine what may be claimed "
    "from results computed on it."
)

KEYWORDS = ("Arabic NLP; fact verification; claim-evidence datasets; dataset "
            "artifacts; annotation quality; evidence grounding")

# ---------------------------------------------------------------- references
# inline: the author-year form; ieee/acl: the full entry in each style.
REFERENCES = {
    "alhindi2021": {
        "inline": "Alhindi et al., 2021",
        "ieee": 'T. Alhindi, A. Alabdulkarim, A. Alshehri, M. Abdul-Mageed, and '
                'P. Nakov, "AraStance: A multi-country and multi-domain dataset '
                'of Arabic stance detection for fact checking," in Proc. NLP4IF, '
                '2021.',
        "acl": "Tariq Alhindi, Amal Alabdulkarim, Ahmed Alshehri, Muhammad "
               "Abdul-Mageed, and Preslav Nakov. 2021. AraStance: A "
               "multi-country and multi-domain dataset of Arabic stance "
               "detection for fact checking. In Proceedings of NLP4IF.",
    },
    "haouari2021": {
        "inline": "Haouari et al., 2021",
        "ieee": 'F. Haouari, M. Hasanain, R. Suwaileh, and T. Elsayed, '
                '"ArCOV19-Rumors: Arabic COVID-19 Twitter dataset for '
                'misinformation detection," in Proc. Arabic NLP Workshop, 2021.',
        "acl": "Fatima Haouari, Maram Hasanain, Reem Suwaileh, and Tamer "
               "Elsayed. 2021. ArCOV19-Rumors: Arabic COVID-19 Twitter dataset "
               "for misinformation detection. In Proceedings of the Arabic NLP "
               "Workshop.",
    },
    "mostafa2024": {
        "inline": "Mostafa and Almogren, 2024",
        "ieee": 'M. A. Mostafa and A. Almogren, "VERA-ARAB: Unveiling the Arabic '
                'tweets credibility by constructing balanced news dataset for '
                'veracity analysis," PeerJ Comput. Sci., vol. 10, e2432, 2024.',
        "acl": "Mohamed A. Mostafa and Ahmad Almogren. 2024. VERA-ARAB: "
               "Unveiling the Arabic tweets credibility by constructing "
               "balanced news dataset for veracity analysis. PeerJ Computer "
               "Science, 10:e2432.",
    },
    "khalil2022": {
        "inline": "Khalil et al., 2022",
        "ieee": 'A. Khalil, M. Jarrah, and M. Aldwairi, "Arabic fake news '
                'dataset (AFND)," Data in Brief, 2022.',
        "acl": "Ashraf Khalil, Mohammad Jarrah, and Monther Aldwairi. 2022. "
               "Arabic fake news dataset (AFND). Data in Brief.",
    },
    "thorne2018": {
        "inline": "Thorne et al., 2018",
        "ieee": 'J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, '
                '"FEVER: A large-scale dataset for fact extraction and '
                'verification," in Proc. NAACL-HLT, 2018.',
        "acl": "James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and "
               "Arpit Mittal. 2018. FEVER: A large-scale dataset for fact "
               "extraction and verification. In Proceedings of NAACL-HLT.",
    },
    "gururangan2018": {
        "inline": "Gururangan et al., 2018",
        "ieee": 'S. Gururangan, S. Swayamdipta, O. Levy, R. Schwartz, S. R. '
                'Bowman, and N. A. Smith, "Annotation artifacts in natural '
                'language inference data," in Proc. NAACL-HLT, 2018.',
        "acl": "Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, "
               "Samuel R. Bowman, and Noah A. Smith. 2018. Annotation artifacts "
               "in natural language inference data. In Proceedings of NAACL-HLT.",
    },
    "poliak2018": {
        "inline": "Poliak et al., 2018",
        "ieee": 'A. Poliak, J. Naradowsky, A. Haldar, R. Rudinger, and B. Van '
                'Durme, "Hypothesis only baselines in natural language '
                'inference," in Proc. *SEM, 2018.',
        "acl": "Adam Poliak, Jason Naradowsky, Aparajita Haldar, Rachel "
               "Rudinger, and Benjamin Van Durme. 2018. Hypothesis only "
               "baselines in natural language inference. In Proceedings of *SEM.",
    },
    "schuster2019": {
        "inline": "Schuster et al., 2019",
        "ieee": 'T. Schuster et al., "Towards debiasing fact verification '
                'models," in Proc. EMNLP, 2019.',
        "acl": "Tal Schuster, Darsh J. Shah, Yun Jie Serene Yeo, Daniel Filizzola, "
               "Enrico Santus, and Regina Barzilay. 2019. Towards debiasing fact "
               "verification models. In Proceedings of EMNLP.",
    },
    "artstein2008": {
        "inline": "Artstein and Poesio, 2008",
        "ieee": 'R. Artstein and M. Poesio, "Inter-coder agreement for '
                'computational linguistics," Comput. Linguistics, vol. 34, no. 4, '
                'pp. 555-596, 2008.',
        "acl": "Ron Artstein and Massimo Poesio. 2008. Inter-coder agreement for "
               "computational linguistics. Computational Linguistics, "
               "34(4):555-596.",
    },
    "cohen1960": {
        "inline": "Cohen, 1960",
        "ieee": 'J. Cohen, "A coefficient of agreement for nominal scales," Educ. '
                'Psychol. Meas., vol. 20, no. 1, pp. 37-46, 1960.',
        "acl": "Jacob Cohen. 1960. A coefficient of agreement for nominal "
               "scales. Educational and Psychological Measurement, 20(1):37-46.",
    },
    "landis1977": {
        "inline": "Landis and Koch, 1977",
        "ieee": 'J. R. Landis and G. G. Koch, "The measurement of observer '
                'agreement for categorical data," Biometrics, vol. 33, no. 1, '
                'pp. 159-174, 1977.',
        "acl": "J. Richard Landis and Gary G. Koch. 1977. The measurement of "
               "observer agreement for categorical data. Biometrics, "
               "33(1):159-174.",
    },
    "efron1993": {
        "inline": "Efron and Tibshirani, 1993",
        "ieee": 'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. '
                'New York, NY, USA: Chapman & Hall, 1993.',
        "acl": "Bradley Efron and Robert J. Tibshirani. 1993. An Introduction to "
               "the Bootstrap. Chapman & Hall, New York.",
    },
    "kish1965": {
        "inline": "Kish, 1965",
        "ieee": 'L. Kish, Survey Sampling. New York, NY, USA: Wiley, 1965.',
        "acl": "Leslie Kish. 1965. Survey Sampling. Wiley, New York.",
    },
    "rashkin2023": {
        "inline": "Rashkin et al., 2023",
        "ieee": 'H. Rashkin et al., "Measuring attribution in natural language '
                'generation models," Comput. Linguistics, vol. 49, no. 4, 2023.',
        "acl": "Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, "
               "Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, "
               "Iulia Turc, and David Reitter. 2023. Measuring attribution in "
               "natural language generation models. Computational Linguistics, "
               "49(4).",
    },
    "honovich2022": {
        "inline": "Honovich et al., 2022",
        "ieee": 'O. Honovich et al., "TRUE: Re-evaluating factual consistency '
                'evaluation," in Proc. NAACL-HLT, 2022.',
        "acl": "Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, "
               "Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, "
               "Avinatan Hassidim, and Yossi Matias. 2022. TRUE: Re-evaluating "
               "factual consistency evaluation. In Proceedings of NAACL-HLT.",
    },
    "shehata2024": {
        "inline": "Shehata et al., 2024",
        "ieee": 'A. M. K. Shehata et al., "ArabFake: A multitask deep learning '
                'framework for Arabic fake news detection," 2024.',
        "acl": "Ahmed M. K. Shehata et al. 2024. ArabFake: A multitask deep "
               "learning framework for Arabic fake news detection.",
    },
    "antoun2020": {
        "inline": "Antoun et al., 2020",
        "ieee": 'W. Antoun, F. Baly, and H. Hajj, "AraBERT: Transformer-based '
                'model for Arabic language understanding," in Proc. OSACT, 2020.',
        "acl": "Wissam Antoun, Fady Baly, and Hazem Hajj. 2020. AraBERT: "
               "Transformer-based model for Arabic language understanding. In "
               "Proceedings of OSACT.",
    },
}

# ---------------------------------------------------------------- body
BODY = [
    ("h1", "Introduction"),
    ("p",
     "Most Arabic misinformation resources label a claim and stop there. A "
     "tweet or headline is marked true or false, and nothing is attached to "
     "say why. A model trained on such data learns to recognise claims that "
     "resemble previously labelled ones, which is a useful signal and a "
     "fragile one: it can be right for reasons that have nothing to do with "
     "whether the claim is supported, and it leaves a reader no trail to "
     "inspect. Evidence-based verification, as formalised for English by FEVER "
     "{c:thorne2018}, asks a different question, namely whether a specific "
     "passage supports, refutes or fails to settle a specific claim."),
    ("p",
     "This paper contributes AFVD-Core, an Arabic claim-evidence dataset built "
     "by harmonising three public Arabic sources with a UAE target-domain "
     "collection and attaching retrieved evidence, provenance, URLs and silver "
     "verification fields, under grouped splits with published fingerprints. "
     "It also contributes something we consider inseparable from the resource: "
     "an audit of what the labels measure."),
    ("p",
     "The audit exists because of how datasets like this one are necessarily "
     "assembled. The claim labels are inherited from fact-checking sources, "
     "where they record whether a claim was true in the world. The evidence is "
     "attached afterwards by retrieval. Nothing in that process guarantees "
     "that the attached passage supports the inherited label, and if it does "
     "not, a dataset that presents itself as evidence-based is measuring "
     "something else. We test this directly rather than assuming either way."),
    ("p",
     "Three audits are reported. A mechanical check asks how often the "
     "evidence is structurally incapable of settling its claim. A provenance "
     "control asks how much of the label is recoverable from the publisher "
     "alone, in the spirit of the hypothesis-only and artifact analyses that "
     "reshaped natural language inference {c:gururangan2018}{c:poliak2018} and "
     "fact verification {c:schuster2019}. A human audit of a stratified sample "
     "asks whether a reader following a written codebook reaches the same "
     "verdicts as the silver labels. We also measure a structural property "
     "that affects every number computed on the dataset: how far repeated "
     "evidence passages reduce the effective sample size of its test sets."),
    ("p",
     "The results are not flattering to the labels, and we report them with "
     "the dataset rather than after it. The dataset remains usable. What "
     "changes is what may be claimed from results computed on it."),

    ("h1", "Related Work"),
    ("p",
     "Arabic misinformation datasets have grown quickly. AraStance "
     "{c:alhindi2021} provides multi-country, multi-domain stance annotation "
     "for fact checking. ArCOV19-Rumors {c:haouari2021} covers COVID-19 "
     "rumours on Arabic Twitter. VERA-ARAB {c:mostafa2024} builds a balanced "
     "credibility corpus of Arabic tweets. AFND {c:khalil2022} labels articles "
     "by publisher credibility rather than by claim, and ArabFake "
     "{c:shehata2024} approaches detection as a multitask problem. Pretrained "
     "Arabic encoders such as AraBERT {c:antoun2020} make strong claim-only "
     "classifiers straightforward to train on any of them."),
    ("p",
     "What these resources largely lack is an evidence side. A claim carries a "
     "verdict but not a passage that justifies it, so a system cannot be asked "
     "to cite, and an evaluation cannot distinguish a verdict reached from "
     "evidence from one reached by recognising the claim. Work on measuring "
     "attribution {c:rashkin2023} and factual consistency {c:honovich2022} "
     "assumes the evidence side exists and is meaningful, which is exactly "
     "what a dataset of this kind must establish rather than assert."),
    ("p",
     "Our audit follows a line of work that treats datasets as objects to be "
     "measured. Annotation artifacts in natural language inference let a model "
     "reach high accuracy from the hypothesis alone {c:gururangan2018}"
     "{c:poliak2018}, and fact verification models were shown to exploit claim-"
     "only cues {c:schuster2019}. The lesson transfers: before reporting what "
     "a model learned, report what could be learned without reading the "
     "evidence at all."),

    ("h1", "AFVD-Core"),
    ("h2", "Sources and harmonisation"),
    ("p",
     "AFVD-Core draws claims from VERA-ARAB, ArCOV19 tweets and claims, and "
     "AraStance, and adds a UAE evidence-enriched target collection. AFND "
     "enters in a corpus role only, as a pool of passages for retrieval, "
     "because its labels describe publisher credibility rather than claim "
     "truth and the two are not interchangeable."),
    ("table", "Sources and their role in AFVD-Core.",
     [["Source", "Binary rows", "Three-class rows", "Role"],
      ["VERA-ARAB", "19,973", "19,973", "Claims and labels"],
      ["ArCOV19-tweets", "3,584", "3,584", "Claims and labels"],
      ["ArCOV19-claims", "138", "138", "Claims and labels"],
      ["AraStance", "1,420", "1,758", "Claims and labels"],
      ["UAE target collection", "5,053", "6,390", "Claims, evidence, metadata"],
      ["AFND", "Evidence only", "Evidence only", "Retrieval corpus"]],
     False),
    ("p",
     "Label harmonisation is task-specific. The binary task keeps Real and "
     "Fake. The three-class task maps supported evidence to Real, refuted "
     "evidence to Fake, and insufficient or unusable evidence to Unverified. "
     "Arabic normalisation is deliberately conservative: diacritics and "
     "tatweel are removed, Alef, Ya and Ta Marbuta variants are unified, and "
     "dialect is preserved rather than translated into Modern Standard Arabic, "
     "since aggressive normalisation erases the regional and sarcastic signals "
     "an error analysis later needs."),
    ("table", "Class distribution across the source and UAE task datasets.",
     [["Dataset and task", "Real", "Fake", "Unverified", "Total"],
      ["Source binary", "11,825", "13,290", "-", "25,115"],
      ["UAE binary", "2,635", "2,418", "-", "5,053"],
      ["Source three-class", "11,825", "13,290", "338", "25,453"],
      ["UAE three-class", "863", "697", "4,830", "6,390"]],
     False),
    ("p",
     "The distribution shows the central difficulty of the three-class "
     "setting. Unverified is 338 rows in the source pool and 4,830 in the UAE "
     "collection, so any model trained on one and evaluated on the other faces "
     "severe label shift, and macro-averaged metrics are necessary to keep the "
     "minority class visible."),

    ("h2", "Splits and leakage control"),
    ("p",
     "Splitting is grouped rather than random. Normalised claim groups are "
     "kept within a single partition, and repeated evidence passages are kept "
     "within a single partition, at 70/10/20. Seventy-two conflicting claim "
     "groups covering 196 rows were removed before splitting. Audit "
     "fingerprints record that no protected claim or evidence group crosses "
     "train, validation and test, and every result computed on the dataset "
     "stores the SHA-256 fingerprint of its test manifest, so two numbers can "
     "be compared only when they were computed on identical rows."),

    ("h1", "Audit 1: Evidence That Cannot Settle Its Claim"),
    ("p",
     "The first audit asks a structural question that needs no judgment. Two "
     "conditions make a passage incapable of settling a claim whatever it "
     "says. The passage may reproduce the claim's own opening text, in which "
     "case it is the claim restated rather than evidence about it. Or it may "
     "be too short to assert anything, a place name or a fragment of the claim."),
    ("table", "Evidence that cannot settle its claim, by corpus and label, in "
              "the combined three-class test set of 6,369 rows.",
     [["Corpus", "Label", "Copies the claim", "Five words or fewer", "Either"],
      ["Source", "Fake", "9.9%", "51.9%", "61.8%"],
      ["Source", "Real", "11.8%", "40.9%", "52.7%"],
      ["Source", "Unverified", "0.0%", "0.0%", "0.0%"],
      ["UAE", "Fake", "0.0%", "2.9%", "2.9%"],
      ["UAE", "Real", "0.0%", "4.0%", "4.0%"],
      ["UAE", "Unverified", "0.0%", "2.7%", "2.7%"]],
     True),
    ("p",
     "Across the combined three-class test set, 45.9% of rows meet one of the "
     "two conditions, rising to 54.3% among rows labelled Real or Fake and to "
     "61.8% of source Fake rows. The problem is concentrated in the source "
     "corpora and is almost absent from the UAE collection, whose evidence was "
     "gathered against the claims rather than retrieved afterwards."),
    ("p",
     "Both counts are lower bounds, because a script can only detect these two "
     "cases. Two further patterns require a reader. Evidence may describe a "
     "different event that shares a place or a person with the claim, which is "
     "common in the UAE rows and invisible to the table above. And for "
     "debunked claims, the retrieved evidence is frequently other posts "
     "spreading the same rumour, which reads as support while supplying none. "
     "The human audit below is where those appear."),
    ("p",
     "The consequence is direct. Where the evidence is a copy of the claim or "
     "a single word, a model given claim and evidence is in effect classifying "
     "the claim alone, and any score it reaches on those rows says nothing "
     "about verification against evidence."),

    ("h1", "Audit 2: What the Publisher Alone Explains"),
    ("p",
     "A majority-class floor is the conventional baseline, and it is a weak "
     "one here. Unverified rows come almost entirely from UAE official and "
     "news feeds, while Real and Fake come from the three source corpora, so "
     "the publisher carries much of the label before any text is read. We "
     "therefore add two controls that read neither claim nor evidence. Each "
     "predicts the majority training label of a single metadata field: the "
     "corpus a row came from, or its publisher."),
    ("table", "Macro F1 of metadata-only controls against the best model in "
              "each setting. In the zero-shot setting the UAE publishers are "
              "unseen in training, so both controls fall back to the majority "
              "class.",
     [["Setting", "Majority", "Corpus", "Publisher", "Best model"],
      ["Zero-shot UAE, binary", "0.3237", "0.3237", "0.3237", "0.5637"],
      ["Zero-shot UAE, three-class", "0.0656", "0.0656", "0.0656", "0.1722"],
      ["Source, binary", "0.3460", "0.3460", "0.5094", "0.9016"],
      ["Source, three-class", "0.2287", "0.2287", "0.3316", "0.7610"],
      ["Combined, binary", "0.3424", "0.4606", "0.5026", "0.8669"],
      ["Combined, three-class", "0.2033", "0.5036", "0.6009", "0.8763"]],
     True),
    ("figure", "How much of each label the publisher alone explains.",
     "control_baselines.png", True),
    ("p",
     "The combined three-class setting is where this matters most. The "
     "publisher alone reaches 0.6009 macro F1 against a best model's 0.8763. "
     "The models therefore add 0.2754 over a provenance shortcut rather than "
     "the 0.6730 the majority floor implies, which is about 40% of the "
     "apparent gain. They plainly learn more than publisher style, and the "
     "honest size of that gain is smaller than the conventional baseline "
     "suggests. We recommend reporting a provenance control alongside the "
     "majority class for any dataset assembled from multiple corpora."),

    ("h1", "Audit 3: A Human Reading of the Evidence"),
    ("h2", "Protocol"),
    ("p",
     "One annotator labelled a stratified sample of 240 rows drawn from the "
     "combined three-class test set, 40 from each cell of the label by corpus "
     "grid, so that the UAE Unverified rows carrying most of the labelling "
     "risk are represented rather than diluted. The annotator saw the claim "
     "and the evidence only, never the silver label, the stratum or the "
     "publisher, and recorded a verdict, an evidence relevance grade, a "
     "faithfulness grade and a confidence mark."),
    ("p",
     "The codebook asks whether the evidence shown settles the claim, not "
     "whether the claim is true, and it resolves in four ordered questions: "
     "whether there is enough evidence to read, whether it is independent of "
     "the claim rather than a repetition of it, whether it describes the same "
     "event, and only then whether it confirms or refutes. The first three can "
     "only yield Unverified. This ordering was added after a first version "
     "counted evidence that repeats a claim as confirming it, which would have "
     "labelled debunked rumours as Real wherever the retrieved evidence was "
     "other posts spreading them. The sample was labelled twice, three days "
     "apart, once under each version."),

    ("h2", "Results"),
    ("p",
     "Agreement with the silver verdicts is at or below chance. On the 220 "
     "rows outside the first block, the second pass agrees on 25.9% of rows "
     "(95% CI 18.6% to 33.0%, bootstrapped over evidence groups "
     "{c:efron1993}), with a kappa {c:cohen1960} of -0.110 (-0.214 to -0.016). "
     "The first pass agrees on 35.9% with a kappa of 0.037 (-0.043 to 0.119). "
     "Both intervals include or fall below chance, so the finding does not "
     "depend on which pass is used."),
    ("table", "Human audit of the silver verdict labels by silver label and "
              "corpus, second pass, 220 scored rows.",
     [["Silver label, corpus", "Rows", "Agreement", "Judged Unverified"],
      ["Fake, UAE", "36", "0.0%", "100.0%"],
      ["Fake, source", "39", "5.1%", "71.8%"],
      ["Real, UAE", "34", "11.8%", "88.2%"],
      ["Real, source", "37", "37.8%", "62.2%"],
      ["Unverified, UAE", "37", "100.0%", "100.0%"],
      ["Unverified, source", "37", "0.0%", "0.0%"]],
     False),
    ("p",
     "The disagreement runs in one direction. Of the rows the silver labels "
     "mark Real or Fake, the annotator judged 80.1% Unverified, including "
     "every UAE row labelled Fake. Of the 71 UAE rows whose evidence the "
     "silver field grades Relevant, 61 were judged Irrelevant, giving a "
     "relevance kappa of -0.023. The two fields are measuring different "
     "things: the silver labels record whether a claim was true, and the audit "
     "records whether the attached evidence shows it."),
    ("p",
     "Labelling the sample twice gives a test-retest estimate of how reliable "
     "the reading is: kappa 0.578 for the verdict (0.451 to 0.697) and 0.560 "
     "for relevance, both moderate on the usual bands {c:landis1977}"
     "{c:artstein2008}. That is an upper bound on what two independent readers "
     "would reach, since one reader brings the same habits to both passes, and "
     "it is the reliability figure available here. A second annotator's sheet "
     "was produced by editing the first annotator's rather than from a blank "
     "one, so it carries no independent agreement and is excluded."),
    ("p",
     "Two consistency checks support the reading. All 41 pairs with five words "
     "of evidence or fewer were judged Unverified, as the codebook requires. "
     "Of the five pairs whose evidence copies the claim, two were judged "
     "Unverified and three were not; scoring those three as the rule requires "
     "moves kappa to -0.124, which changes nothing. Human faithfulness agrees "
     "with the silver faithfulness field at kappa 0.362, the only one of the "
     "three fields to show any agreement at all."),

    ("h1", "A Structural Property: Effective Sample Size"),
    ("p",
     "Evidence passages repeat across AFVD-Core, one of them 1,009 times. "
     "Grouped splitting keeps a repeated passage inside a single partition, so "
     "there is no leakage, but it does not make the rows within a partition "
     "independent. Rows sharing a passage are near-duplicates, and an interval "
     "computed by resampling rows is therefore too narrow."),
    ("figure", "Rows against Kish effective sample size for each test set.",
     "effective_sample.png", True),
    ("p",
     "Kish effective sample size {c:kish1965} over evidence clusters makes the "
     "structure visible: the zero-shot UAE test set has 6,390 rows but about "
     "26 effective clusters, and the combined three-class test set has 6,369 "
     "rows and about 319. That figure is a worst-case reading, since a few "
     "very large clusters dominate it. Resampling evidence groups directly "
     "gives the number to report, and widens 95% intervals on macro F1 by 1.22 "
     "to 2.77 times relative to a row-level bootstrap. In the combined "
     "three-class setting the interval spans 0.042, which is wider than most "
     "differences between models we have measured on this data."),

    ("h1", "Discussion"),
    ("p",
     "Taken together the three audits describe a dataset whose claim side is "
     "sound and whose evidence side is weaker than its schema suggests. The "
     "labels are inherited claim-truth judgments. The evidence was attached by "
     "retrieval, and for more than half the committed rows it cannot support "
     "the label it sits beside. A human reading of the evidence disagrees with "
     "the labels at chance. None of this makes the resource unusable, and we "
     "would rather ship it with the measurements than without them."),
    ("p",
     "For anyone assembling a comparable resource, three practices follow from "
     "this work and cost little. Report a provenance control beside the "
     "majority class, since a metadata lookup here recovers most of the "
     "apparent headroom. Run the mechanical evidence checks before release, "
     "since copied and one-word evidence is detectable without annotation. "
     "And audit a stratified sample against a written codebook that asks "
     "whether the evidence settles the claim, since that question is not the "
     "one the inherited labels answer."),
    ("p",
     "For users of AFVD-Core, the practical guidance is to treat scores on the "
     "binary and three-class tasks as claim classification with an evidence "
     "channel of uneven quality, to prefer the UAE portion when the evidence "
     "side matters, and to compute intervals over evidence groups rather than "
     "rows."),

    ("h1", "Limitations"),
    ("p",
     "The human audit rests on one annotator. Its reliability is bounded by a "
     "test-retest kappa of 0.578 rather than by agreement between two people, "
     "and a second independent annotator remains the clearest improvement "
     "available. The audit covers 240 rows, which fixes the agreement rate to "
     "within about 7 points at 95% confidence; more rows would tighten that "
     "estimate without changing a conclusion this far from chance."),
    ("p",
     "The mechanical checks detect two failure modes and miss two others that "
     "need a reader. The silver faithfulness and relevance fields are "
     "themselves unvalidated, which is why they appear here as objects of "
     "measurement rather than as a reference. The UAE collection is one target "
     "domain and does not represent every Arabic dialect, country or media "
     "ecosystem."),

    ("h1", "Ethical Considerations"),
    ("p",
     "A misinformation resource can cause harm by fixing a false label on "
     "truthful speech, by amplifying unreliable publishers, or by presenting "
     "an uncertain decision as settled. AFVD-Core retains evidence URLs and "
     "source names for provenance and audit, and no personal data beyond what "
     "the public source datasets already contain. Unverified must be read as "
     "insufficient support rather than as an accusation of falsehood. The "
     "audits in this paper are published precisely so that claims made from "
     "the dataset can be calibrated to what its labels actually measure."),

    ("h1", "Conclusion"),
    ("p",
     "We presented AFVD-Core, an Arabic claim-evidence verification dataset "
     "with provenance and silver verification metadata under audited grouped "
     "splits, and three audits of what its labels measure. More than half the "
     "rows labelled Real or Fake carry evidence that cannot settle them, a "
     "publisher-only control recovers 0.6009 macro F1 where the best model "
     "reaches 0.8763, and a human reading of the evidence agrees with the "
     "silver verdicts no better than chance. Repeated evidence passages "
     "further reduce the effective size of the test sets. The dataset is "
     "released with these numbers attached, because they are what determines "
     "the claims it can support."),
]
