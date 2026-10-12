import { createContext, useContext, useMemo, useState } from 'react';

// Arabic-is-primary UI language, with an English toggle. This dictionary
// covers chrome/labels only — actual claim text / explanation / evidence
// text keep whatever direction the underlying content itself is in,
// detected independently (see detectDir below).
const STRINGS = {
  brandName: { ar: 'أمان سنتينل', en: 'AMAN Sentinel' },
  brandSub: { ar: 'التحقق من الأخبار بالذكاء الاصطناعي', en: 'AI-Assisted Misinformation Verification' },
  howItWorks: { ar: 'كيف يعمل أمان؟', en: 'How AMAN Works' },
  langToggle: { ar: 'English', en: 'العربية' },

  heroTitleAr: { ar: 'هل يمكنني الوثوق بما وصلني؟', en: 'هل يمكنني الوثوق بما وصلني؟' },
  heroTitleEn: { ar: 'Can I trust what I just received?', en: 'Can I trust what I just received?' },
  heroBody: {
    ar: 'ألصق أي خبر أو رسالة متداولة أدناه. يفحصها أمان أولًا بمصنّفات خفيفة وفورية، ولا يستدعي نموذجًا لغويًا ويبحث عن أدلة حقيقية إلا عند الشك أو عدم الوضوح.',
    en: 'Paste any claim, headline, or forwarded message below. AMAN checks it first with fast, lightweight classifiers — and only calls in evidence retrieval and a grounded LLM when the claim is genuinely uncertain or unfamiliar.',
  },

  inputLabel: { ar: 'النص المطلوب فحصه', en: 'Text to verify' },
  inputPlaceholder: {
    ar: 'ألصق الخبر أو الرسالة أو المنشور هنا… (عربي أو إنجليزي أو مزيج منهما)',
    en: 'Paste the claim, message, or post here… (Arabic, English, or mixed)',
  },
  analyzeBtn: { ar: 'تحليل', en: 'Analyze' },
  analyzingBtn: { ar: 'جارٍ التحليل…', en: 'Analyzing…' },
  clearBtn: { ar: 'مسح', en: 'Clear' },
  emptyHint: {
    ar: 'لم يتم إدخال أي نص بعد. جرّب أحد الأمثلة أدناه أو ألصق نصك الخاص.',
    en: 'Nothing entered yet. Try one of the examples below, or paste your own text.',
  },

  examplesTitle: { ar: 'أمثلة حقيقية من بيانات الاختبار', en: 'Real examples from the test data' },
  examplesSub: {
    ar: 'انقر لتعبئة الحقل أعلاه — ثم اضغط تحليل. السلوك الفعلي يُحدَّد لحظة التشغيل من الواجهة الخلفية، وليس مُلفَّقًا هنا.',
    en: 'Click to prefill the field above, then press Analyze. Actual behavior is decided live by the backend, not scripted here.',
  },

  progressTitle: { ar: 'أمان يحلل النص الآن', en: 'AMAN is analyzing your text' },
  progressElapsed: { ar: 'الوقت المنقضي', en: 'Elapsed' },
  progressWorking: {
    ar: 'جارٍ العمل — هذه الخطوات تقديرية إلى أن يصل الرد الفعلي',
    en: 'Working — these steps are provisional until the real response arrives',
  },
  progressLongWait: {
    ar: 'إذا كان النص غير مألوف أو مثيرًا للشك، قد يستدعي أمان استرجاع أدلة حقيقية وتحقيقًا عميقًا بنموذج لغوي، وهذا قد يستغرق حتى دقيقتين تقريبًا. هذا طبيعي ولا يعني وجود خلل.',
    en: 'If the text is unfamiliar or disputed, AMAN may retrieve real evidence and run deep LLM verification — this can take up to ~2 minutes. That is expected, not a malfunction.',
  },
  progressConfirmedTitle: {
    ar: 'هذه الخطوات التي أكّدت الواجهة الخلفية أنها نُفِّذت فعليًا:',
    en: 'These are the steps the backend confirmed actually ran:',
  },
  viewResultBtn: { ar: 'عرض النتيجة', en: 'View result' },

  errorTitle: { ar: 'حدث خطأ أثناء التحليل', en: 'Something went wrong during analysis' },
  retryBtn: { ar: 'حاول مرة أخرى', en: 'Try again' },

  newAnalysisBtn: { ar: 'تحليل نص جديد', en: 'Analyze new text' },
  resultTitle: { ar: 'نتيجة التحليل', en: 'Analysis result' },

  tierClassifierOnly: { ar: 'فحص تصنيف سريع (دون أدلة)', en: 'Quick classifier check (no evidence)' },
  tierEvidenceGrounded: { ar: 'تحقق مستند إلى أدلة حقيقية', en: 'Evidence-grounded verification' },

  likelyReliable: { ar: 'يُحتمل أنه موثوق', en: 'LIKELY RELIABLE' },
  likelyMisleading: { ar: 'يُحتمل أنه مضلل', en: 'LIKELY MISLEADING' },
  verdictSupported: { ar: 'مدعوم بالأدلة', en: 'SUPPORTED' },
  verdictContradicted: { ar: 'تناقضه الأدلة', en: 'CONTRADICTED' },
  verdictInsufficient: { ar: 'أدلة غير كافية', en: 'INSUFFICIENT EVIDENCE' },

  classifierCaveat: {
    ar: 'هذا تقدير من مصنّف إحصائي سريع فقط — لم تُفحص أي أدلة. ليست إفادة بأن الخبر "صحيح" أو "كاذب" بشكل قاطع.',
    en: 'This is a fast statistical-classifier estimate only — no evidence was checked. It is not a definitive claim that the news is "true" or "false".',
  },
  evidenceCaveat: {
    ar: 'هذه النتيجة مستندة إلى أدلة فعلية استُرجعت ثم استنتج منها نموذج لغوي — وهي إفادة أقوى من فحص المصنّف وحده.',
    en: 'This result is grounded in retrieved evidence that a language model reasoned over — a stronger claim than the classifier-only path.',
  },

  confidenceLabel: { ar: 'مستوى الثقة', en: 'Confidence' },
  noveltyLabel: { ar: 'درجة الجدة', en: 'Novelty' },
  noveltyFamiliar: { ar: 'مألوف', en: 'FAMILIAR' },
  noveltyModerate: { ar: 'جديد نسبيًا', en: 'MODERATELY NOVEL' },
  noveltyHigh: { ar: 'جديد جدًا', en: 'HIGHLY NOVEL' },
  noveltyExplain: {
    ar: 'تقيس هذه الدرجة مدى اختلاف صياغة النص أو موضوعه عمّا رآه أمان سابقًا. لا تعني هذه الدرجة أن الخبر كاذب أو صحيح — فقط أنه مألوف أو غير مألوف.',
    en: "This measures how different this text's wording/topic is from what AMAN has seen before. It does not mean the claim is fake or real — only unfamiliar or familiar.",
  },

  agreementLabel: { ar: 'توافق النماذج الداخلية', en: 'Internal model agreement' },
  agreementAll: { ar: 'اتفقت النماذج الثلاثة جميعها', en: 'All 3 internal models agreed' },
  agreementSplit: { ar: 'انقسمت النماذج 2 إلى 1', en: 'Models split 2-to-1' },

  deepVerifLabel: { ar: 'التحقق العميق', en: 'Deep verification' },
  deepVerifUsedYes: { ar: 'نعم، استُخدم', en: 'Used' },
  deepVerifUsedNo: { ar: 'لا، لم يُستخدم', en: 'Not used' },
  deepVerifWhySplit: {
    ar: 'اختلفت المصنّفات في رأيها، فبحث أمان عن أدلة حقيقية قبل الإجابة.',
    en: 'The classifiers disagreed with each other, so AMAN checked real evidence before answering.',
  },
  deepVerifWhyNovel: {
    ar: 'كانت صياغة النص غير مألوفة للمصنّفات، فبحث أمان عن أدلة حقيقية قبل الإجابة.',
    en: "This claim's wording was unfamiliar to AMAN's classifiers, so AMAN checked real evidence before answering.",
  },
  deepVerifWhyLowConf: {
    ar: 'لم تكن ثقة المصنّفات كافية للإجابة الفورية، فبحث أمان عن أدلة حقيقية قبل الإجابة.',
    en: "AMAN's confidence wasn't high enough to answer automatically, so it checked real evidence before answering.",
  },
  deepVerifWhyNo: {
    ar: 'اتفقت المصنّفات بثقة عالية على نص مألوف، فلم تكن هناك حاجة للبحث عن أدلة أو استدعاء نموذج لغوي.',
    en: "AMAN's classifiers agreed with high confidence on familiar text, so no evidence lookup or LLM call was needed.",
  },

  explanationLabel: { ar: 'توضيح أمان', en: "AMAN's explanation" },
  evidenceLabel: { ar: 'الأدلة المستشهد بها', en: 'Cited evidence' },
  evidenceSource: { ar: 'المصدر', en: 'Source' },
  evidenceMethod: { ar: 'طريقة الاسترجاع', en: 'Retrieval method' },
  evidenceScore: { ar: 'درجة الاسترجاع', en: 'Retrieval score' },
  noEvidence: {
    ar: 'لم يُسترجع أي دليل لهذا النص — لم تكن هناك حاجة لذلك في هذا المسار.',
    en: 'No evidence was retrieved for this text — it was not needed on this path.',
  },

  techDetailsToggle: { ar: 'تفاصيل تقنية', en: 'Technical details' },
  trustScore: { ar: 'درجة الثقة الموحّدة', en: 'Trust score' },
  routerThreshold: { ar: 'عتبة التوجيه', en: 'Router threshold' },
  processingTime: { ar: 'زمن المعالجة', en: 'Processing time' },
  seconds: { ar: 'ثانية', en: 's' },

  modalTitle: { ar: 'كيف يعمل أمان سنتينل؟', en: 'How AMAN Sentinel Works' },
  modalClose: { ar: 'إغلاق', en: 'Close' },
  modalIntro: {
    ar: 'أمان لا يستدعي نموذجًا لغويًا لكل نص. أولًا، تفحص ثلاث مصنّفات خفيفة (الأساسي SVM وآخران لاكتشاف الخلاف) النص مع كاشف للحالات غير المألوفة. تُحسم معظم الحالات فورًا دون أي نموذج لغوي. فقط الحالات غير المؤكدة أو غير المألوفة أو المتنازع عليها تُصعَّد لاسترجاع أدلة حقيقية (BM25 + بحث دلالي + دمج هجين) ثم نموذج لغوي يُحاجج فقط بالفقرات المسترجعة، ويجب أن يستشهد بالأدلة أو يمتنع عن الإجابة.',
    en: 'AMAN does not call an LLM for every input. First, three lightweight classifiers (a primary SVM plus two others used only to detect disagreement) screen the text, alongside a novelty/out-of-distribution detector. Most cases are resolved immediately with no LLM involved. Only genuinely uncertain, unfamiliar, or disputed claims escalate to real evidence retrieval (BM25 + dense + hybrid-RRF) followed by an LLM that reasons only over retrieved passages and must cite evidence or abstain.',
  },
  modalClassifierHeading: { ar: 'أداء المصنّف الأساسي (SVM)', en: 'Primary classifier (SVM) performance' },
  modalAgreementHeading: { ar: 'لماذا يُصعَّد الاختلاف؟', en: 'Why disagreement triggers escalation' },
  modalAgreementBody: {
    ar: 'عندما تتفق النماذج الداخلية الثلاثة (67% من الحالات)، تبلغ الدقة 92.6%. وعندما تنقسم 2 مقابل 1 (33% من الحالات)، تنخفض الدقة إلى 64.9% فقط — وهذا هو السبب الدقيق الذي يجعل الاختلاف بين النماذج مُحفِّزًا للتصعيد.',
    en: 'When all 3 internal models agree (67% of cases), accuracy is 92.6%. When they split 2-1 (33% of cases), accuracy drops to just 64.9% — this is precisely why model disagreement triggers escalation.',
  },
  modalSelectiveHeading: { ar: 'نقطة التشغيل المختارة (تغطية 80%)', en: 'Selected operating point (80% coverage)' },
  modalSelectiveBody: {
    ar: 'عند نقطة التشغيل هذه: تغطية 76.90% بدقة انتقائية 92.65% لما تمت تغطيته.',
    en: 'At this operating point: 76.90% coverage at 92.65% selective accuracy for what is covered.',
  },
  modalRetrievalHeading: { ar: 'أداء الاسترجاع', en: 'Retrieval performance' },
  modalRetrievalDisclosure: {
    ar: 'تنبيه مهم: أرقام أداء الاسترجاع أعلاه قيست فقط على جزء من البيانات (مصدر واحد) كان فيه مرجع أدلة قابل لإعادة الاستخدام داخل مجموعة الاختبار. مصادر البيانات الأخرى لم يكن فيها أي مرجع أدلة قابل لإعادة الاستخدام داخل المجموعة على الإطلاق، لذا لم تُقَس جودة الاسترجاع لها.',
    en: 'Important disclosure: the retrieval numbers above were only measurable on a subset of the data (one source corpus) where reusable reference evidence existed in the test corpus. The other source corpora had no reusable in-corpus reference evidence at all, so retrieval quality for those is not measured.',
  },
};

export const LangContext = createContext(null);

export function LangProvider({ children }) {
  const [lang, setLang] = useState('ar');
  const value = useMemo(() => {
    const t = (key) => STRINGS[key]?.[lang] ?? key;
    return {
      lang,
      setLang,
      toggleLang: () => setLang((l) => (l === 'ar' ? 'en' : 'ar')),
      t,
      dir: lang === 'ar' ? 'rtl' : 'ltr',
    };
  }, [lang]);
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang() {
  const ctx = useContext(LangContext);
  if (!ctx) throw new Error('useLang must be used within LangProvider');
  return ctx;
}

// Detect direction of arbitrary content text (claim input, explanation,
// evidence passages) independently from the UI chrome language — per the
// brief, the input/output should drive its own RTL/LTR, not the UI toggle.
const ARABIC_RANGE = /[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]/;
export function detectDir(text) {
  if (!text) return 'rtl'; // default Arabic-first
  return ARABIC_RANGE.test(text) ? 'rtl' : 'ltr';
}
