import { useEffect, useState } from 'react';
import { useLang } from '../i18n.jsx';

// Labels for every stage name the backend's processing_stages array can
// contain. The first 4 are the "always runs" fast path; the last 3 only
// appear for escalated requests (retrieving_evidence, deep_verification)
// or at the very end (preparing_result).
const STAGE_LABELS = {
  analyzing_language: { ar: 'تحليل اللغة', en: 'Analyzing language' },
  running_classifiers: { ar: 'تشغيل المصنّفات', en: 'Running classifiers' },
  checking_confidence: { ar: 'فحص مستوى الثقة', en: 'Checking confidence' },
  checking_novelty: { ar: 'فحص درجة الجدة', en: 'Checking novelty' },
  retrieving_evidence: { ar: 'استرجاع الأدلة', en: 'Retrieving evidence' },
  deep_verification: { ar: 'تحقق عميق بالنموذج اللغوي', en: 'Deep verification (LLM)' },
  preparing_result: { ar: 'تجهيز النتيجة', en: 'Preparing result' },
};

// Optimistic/generic sequence shown while we don't yet know the real
// processing_stages (the backend does not stream). This is NOT a claim
// about what has actually run — the UI makes that explicit, and reconciles
// with the real list the moment the response lands.
const OPTIMISTIC_SEQUENCE = [
  'analyzing_language',
  'running_classifiers',
  'checking_confidence',
  'checking_novelty',
];

function formatElapsed(ms) {
  const totalSeconds = Math.floor(ms / 1000);
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

export default function AnalysisProgress({ elapsedMs, finalStages, onContinue }) {
  const { t, lang } = useLang();
  const [pulseIndex, setPulseIndex] = useState(0);

  const pending = finalStages === null;

  useEffect(() => {
    if (!pending) return;
    const id = setInterval(() => {
      setPulseIndex((i) => (i + 1) % OPTIMISTIC_SEQUENCE.length);
    }, 1300);
    return () => clearInterval(id);
  }, [pending]);

  // Once the real stage list arrives, hold it on screen briefly (confirmed
  // ticks are satisfying feedback) then hand off to the result dashboard.
  useEffect(() => {
    if (pending) return;
    const id = setTimeout(() => onContinue(), 900);
    return () => clearTimeout(id);
  }, [pending, onContinue]);

  const showLongWaitNote = pending && elapsedMs > 6000;

  return (
    <div className="max-w-xl mx-auto px-4 sm:px-6 py-16 aman-fade-up">
      <div className="text-center mb-8">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl mb-4 text-2xl aman-pulse-dot" style={{ background: 'var(--aman-teal-dim)' }}>
          🛰️
        </div>
        <h1 className="text-xl font-extrabold mb-1" style={{ color: 'var(--aman-text)' }}>
          {t('progressTitle')}
        </h1>
        <p className="text-sm" style={{ color: 'var(--aman-text-dim)' }}>
          {t('progressElapsed')}: <span className="font-mono font-semibold">{formatElapsed(elapsedMs)}</span>
        </p>
      </div>

      <div
        className="rounded-2xl border p-5"
        style={{ background: 'var(--aman-bg-card)', borderColor: 'var(--aman-border)' }}
      >
        {pending ? (
          <>
            <p className="text-xs font-semibold mb-4" style={{ color: 'var(--aman-amber)' }}>
              {t('progressWorking')}
            </p>
            <ol className="space-y-3">
              {OPTIMISTIC_SEQUENCE.map((stageKey, idx) => {
                const active = idx === pulseIndex;
                return (
                  <li key={stageKey} className="flex items-center gap-3">
                    <span
                      className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${active ? 'aman-pulse-dot' : ''}`}
                      style={{ background: active ? 'var(--aman-teal)' : 'var(--aman-border)' }}
                    />
                    <span
                      className="text-sm"
                      style={{ color: active ? 'var(--aman-text)' : 'var(--aman-text-dim)' }}
                    >
                      {STAGE_LABELS[stageKey][lang]}
                    </span>
                  </li>
                );
              })}
            </ol>

            {showLongWaitNote && (
              <div
                className="mt-5 pt-4 border-t text-xs leading-relaxed aman-shimmer rounded-lg p-3"
                style={{ borderColor: 'var(--aman-border)', color: 'var(--aman-text-dim)' }}
              >
                {t('progressLongWait')}
              </div>
            )}
          </>
        ) : (
          <>
            <p className="text-xs font-semibold mb-4" style={{ color: 'var(--aman-teal)' }}>
              {t('progressConfirmedTitle')}
            </p>
            <ol className="space-y-3">
              {finalStages.map((stageKey) => (
                <li key={stageKey} className="flex items-center gap-3">
                  <span
                    className="w-5 h-5 rounded-full flex items-center justify-center text-[11px] flex-shrink-0"
                    style={{ background: 'var(--aman-teal-dim)', color: 'var(--aman-teal)' }}
                  >
                    ✓
                  </span>
                  <span className="text-sm font-medium" style={{ color: 'var(--aman-text)' }}>
                    {STAGE_LABELS[stageKey]?.[lang] || stageKey}
                  </span>
                </li>
              ))}
            </ol>
            <button
              onClick={onContinue}
              className="mt-5 w-full text-center py-2.5 rounded-xl font-bold text-sm"
              style={{ background: 'var(--aman-teal)', color: '#06211d' }}
            >
              {t('viewResultBtn')}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
