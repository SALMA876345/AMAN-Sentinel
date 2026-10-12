// ============================================================================
// DEV FIXTURE — NOT REAL BACKEND OUTPUT.
//
// This file is only used for local frontend-only development when
// VITE_API_URL is NOT configured (see src/api.js). The moment a real
// backend URL is set, this file is never called. None of the JSON below
// came from a live AMAN pipeline run — it exists purely so the UI can be
// built/tested before the FastAPI backend is reachable. Do not treat this
// as ground truth and do not let it leak into any screen as if it were a
// real verification result.
// ============================================================================

import { demoExamples } from './data/demoExamples';

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

const FAST_STAGES = [
  'analyzing_language',
  'running_classifiers',
  'checking_confidence',
  'checking_novelty',
  'preparing_result',
];

const ESCALATED_STAGES = [
  'analyzing_language',
  'running_classifiers',
  'checking_confidence',
  'checking_novelty',
  'retrieving_evidence',
  'deep_verification',
  'preparing_result',
];

function fastFixture(classification, confidence) {
  return {
    status: 'VERIFIED_AUTOMATICALLY',
    classification,
    confidence,
    novelty: { level: 'FAMILIAR', score: 0.14 },
    model_agreement: 'ALL_AGREE',
    deep_verification_used: false,
    evidence_verdict: null,
    explanation:
      classification === 'Fake'
        ? '[DEV FIXTURE] اتفقت المصنّفات الثلاثة بثقة عالية على أن نمط هذا النص يشبه أنماط الأخبار المضللة التي رآها أمان سابقًا.'
        : '[DEV FIXTURE] اتفقت المصنّفات الثلاثة بثقة عالية على أن نمط هذا النص يشبه أنماط الأخبار الموثوقة التي رآها أمان سابقًا.',
    evidence: [],
    processing_stages: FAST_STAGES,
    metadata: {
      trust_score: confidence,
      router_threshold: 0.8,
      model_votes: [classification, classification, classification],
      processing_time_seconds: 1.3,
    },
  };
}

function escalatedSupportedFixture() {
  return {
    status: 'VERIFIED_WITH_EVIDENCE',
    classification: null,
    confidence: 0.55,
    novelty: { level: 'MODERATELY NOVEL', score: 0.48 },
    model_agreement: 'PARTIAL_AGREEMENT',
    deep_verification_used: true,
    evidence_verdict: 'SUPPORTED',
    explanation:
      '[DEV FIXTURE] اختلفت المصنّفات في رأيها حول هذا النص، فاسترجع أمان فقرات ذات صلة ووجد فيها نموذج لغوي ما يدعم مضمون الخبر، مع استشهاد صريح بالمصدر.',
    evidence: [
      {
        id: 'dev-fixture-ev-1',
        text: 'القبض على مجموعة يديرون مصنع يحول زيت السيارات إلى عسل يصنع في السعوديه. الدخان الجديد مغشوش31 القبض على مصنع يحول زيت السيارات الى عسل يصنع في السعوديه',
        source: '[DEV FIXTURE] corpus_passage_19553',
        retrieval_method: 'hybrid_rrf',
        retrieval_score: 0.81,
      },
    ],
    processing_stages: ESCALATED_STAGES,
    metadata: {
      trust_score: 0.55,
      router_threshold: 0.8,
      model_votes: ['Real', 'Fake', 'Real'],
      processing_time_seconds: 34.2,
    },
  };
}

function escalatedInsufficientFixture() {
  return {
    status: 'UNCERTAIN',
    classification: null,
    confidence: 0.5,
    novelty: { level: 'HIGHLY NOVEL', score: 0.72 },
    model_agreement: 'PARTIAL_AGREEMENT',
    deep_verification_used: true,
    evidence_verdict: 'INSUFFICIENT_EVIDENCE',
    explanation:
      '[DEV FIXTURE] نجح استرجاع الأدلة لكن إجابة النموذج اللغوي لم تجتز ضمانة الاستشهاد الصريح بالأدلة، فخفّض أمان النتيجة إلى "أدلة غير كافية" عمدًا بدل التخمين.',
    evidence: [
      {
        id: 'dev-fixture-ev-2',
        text: 'رصد استاد برج العرب يتحول لـمسبح برج العرب.. والمدير: لا أملك حق التصريح',
        source: '[DEV FIXTURE] corpus_passage_6985',
        retrieval_method: 'bm25',
        retrieval_score: 0.19,
      },
    ],
    processing_stages: ESCALATED_STAGES,
    metadata: {
      trust_score: 0.5,
      router_threshold: 0.8,
      model_votes: ['Real', 'Real', 'Fake'],
      processing_time_seconds: 41.7,
    },
  };
}

function genericFixture(text) {
  const hasArabic = /[؀-ۿ]/.test(text);
  const looksShort = text.trim().length < 40;
  const classification = looksShort ? 'Fake' : 'Real';
  return {
    ...fastFixture(classification, looksShort ? 0.88 : 0.91),
    explanation: hasArabic
      ? '[DEV FIXTURE] استجابة تجريبية محلية — لا يوجد اتصال حقيقي بالواجهة الخلفية. اضبط VITE_API_URL لرؤية نتائج حقيقية.'
      : '[DEV FIXTURE] Local placeholder response — no real backend connected. Set VITE_API_URL to see real results.',
  };
}

export async function mockVerify(text, _evidenceText = '') {
  const trimmed = (text || '').trim();
  const match = demoExamples.find(
    (ex) => trimmed && (trimmed.includes(ex.claim) || ex.claim.includes(trimmed))
  );

  if (match?.label === 'high_confidence_no_escalation') {
    await wait(1400);
    return fastFixture('Fake', 0.94);
  }
  if (match?.label === 'escalation_with_evidence') {
    await wait(4500);
    return escalatedSupportedFixture();
  }
  if (match?.label === 'insufficient_evidence') {
    await wait(4000);
    return escalatedInsufficientFixture();
  }

  await wait(1200);
  return genericFixture(trimmed);
}

export async function mockHealth() {
  await wait(150);
  return { status: 'ok', pipeline_loaded: true };
}
