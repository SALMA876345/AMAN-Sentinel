// Real demo claims sourced verbatim from phase4/demo_examples.json
// (hackathon/data/clean_split/source_binary_test.csv rows). Do NOT invent new
// example claims here — these three exist specifically to demonstrate the
// three real pipeline paths (fast trusted-path, escalation+evidence,
// escalation+safeguard-downgrade). The actual verdict for each is produced
// live by the backend at request time, not hard-coded here.

export const demoExamples = [
  {
    id: 'CLEAN-000123-5f9bb76b0f',
    label: 'high_confidence_no_escalation',
    tag: { ar: 'مسار سريع', en: 'Fast path' },
    description: {
      ar: 'يتوقع أن تتفق المصنّفات الثلاثة بثقة عالية، فتُرجَع نتيجة فورية دون أي نموذج لغوي.',
      en: 'Expected: all 3 classifiers agree with high confidence — instant answer, no LLM involved.',
    },
    claim:
      'الاسطوره الايفواريه ديديه دروجبا يعتنق الإسلام ،،،،،اللهم ثبته علي الحق الذي يرضيك',
  },
  {
    id: 'CLEAN-019553-27c98aedd4',
    label: 'escalation_with_evidence',
    tag: { ar: 'تصعيد + أدلة', en: 'Escalation + evidence' },
    description: {
      ar: 'يتوقع أن تتعارض المصنّفات أو تكون ثقتها منخفضة، فيتم التصعيد لاسترجاع الأدلة والتحقق العميق.',
      en: 'Expected: classifiers disagree or are low-confidence — escalates to evidence retrieval + deep verification.',
    },
    claim: 'شاهد: ضبط مصنع يحول زيت السيارات إلى عسل',
  },
  {
    id: 'CLEAN-006985-1aa60ddff7',
    label: 'insufficient_evidence',
    tag: { ar: 'أدلة غير كافية', en: 'Insufficient evidence' },
    description: {
      ar: 'يتوقع تصعيدًا تنجح فيه عملية الاسترجاع، لكن إجابة النموذج لا تجتاز ضمانة الاستشهاد بالأدلة.',
      en: 'Expected: escalates, retrieval succeeds, but the LLM answer fails the citation safeguard and gets downgraded.',
    },
    claim: 'استاد برج العرب غرق بمياه الامطار',
  },
];
