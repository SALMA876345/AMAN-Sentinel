import { useState } from 'react';
import { useLang, detectDir } from '../i18n.jsx';
import StatusBadge, { classifyResult } from './StatusBadge.jsx';
import ConfidenceMeter from './ConfidenceMeter.jsx';
import NoveltyBadge from './NoveltyBadge.jsx';
import EvidenceList from './EvidenceList.jsx';

function deepVerifReasonKey(result) {
  if (!result.deep_verification_used) return 'deepVerifWhyNo';
  if (result.model_agreement === 'PARTIAL_AGREEMENT') return 'deepVerifWhySplit';
  if (result.novelty?.level === 'HIGHLY NOVEL') return 'deepVerifWhyNovel';
  return 'deepVerifWhyLowConf';
}

function Card({ children }) {
  return (
    <div
      className="rounded-2xl border p-5"
      style={{ background: 'var(--aman-bg-card)', borderColor: 'var(--aman-border)' }}
    >
      {children}
    </div>
  );
}

export default function ResultDashboard({ result, inputText, onReset }) {
  const { t, lang } = useLang();
  const [showTech, setShowTech] = useState(false);
  const info = classifyResult(result);
  const hasEvidence = Array.isArray(result.evidence) && result.evidence.length > 0;

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-10 space-y-5 aman-fade-up">
      <div className="flex items-center justify-between gap-3">
        <h1 className="text-lg font-extrabold" style={{ color: 'var(--aman-text)' }}>
          {t('resultTitle')}
        </h1>
        <button
          onClick={onReset}
          className="px-4 py-2 rounded-xl font-semibold text-sm border"
          style={{ borderColor: 'var(--aman-border)', color: 'var(--aman-text-dim)' }}
        >
          {t('newAnalysisBtn')}
        </button>
      </div>

      {inputText && (
        <Card>
          <p
            dir={detectDir(inputText)}
            className="text-sm leading-relaxed"
            style={{ color: 'var(--aman-text-dim)' }}
          >
            “{inputText}”
          </p>
        </Card>
      )}

      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
        <StatusBadge result={result} />
      </div>
      <p className="text-xs leading-relaxed -mt-1" style={{ color: 'var(--aman-text-dim)' }}>
        {t(info.tier === 'classifier-only' ? 'classifierCaveat' : 'evidenceCaveat')}
      </p>

      <div className="grid sm:grid-cols-2 gap-4">
        <Card>
          <ConfidenceMeter confidence={result.confidence} />
        </Card>
        <Card>
          <NoveltyBadge novelty={result.novelty} />
        </Card>
      </div>

      <div className="grid sm:grid-cols-2 gap-4">
        <Card>
          <p className="text-sm font-semibold mb-1.5" style={{ color: 'var(--aman-text-dim)' }}>
            {t('agreementLabel')}
          </p>
          <p className="text-sm font-bold" style={{ color: 'var(--aman-text)' }}>
            {result.model_agreement === 'ALL_AGREE' ? t('agreementAll') : t('agreementSplit')}
          </p>
        </Card>
        <Card>
          <p className="text-sm font-semibold mb-1.5" style={{ color: 'var(--aman-text-dim)' }}>
            {t('deepVerifLabel')}
          </p>
          <p
            className="text-sm font-bold mb-1.5"
            style={{ color: result.deep_verification_used ? 'var(--aman-blue)' : 'var(--aman-text)' }}
          >
            {result.deep_verification_used ? t('deepVerifUsedYes') : t('deepVerifUsedNo')}
          </p>
          <p className="text-xs leading-relaxed" style={{ color: 'var(--aman-text-dim)' }}>
            {t(deepVerifReasonKey(result))}
          </p>
        </Card>
      </div>

      <Card>
        <p className="text-sm font-semibold mb-2" style={{ color: 'var(--aman-text-dim)' }}>
          {t('explanationLabel')}
        </p>
        <p
          dir={detectDir(result.explanation)}
          className="text-sm leading-relaxed"
          style={{ color: 'var(--aman-text)' }}
        >
          {result.explanation || '—'}
        </p>
      </Card>

      {hasEvidence && (
        <Card>
          <p className="text-sm font-semibold mb-3" style={{ color: 'var(--aman-text-dim)' }}>
            {t('evidenceLabel')}
          </p>
          <EvidenceList evidence={result.evidence} />
        </Card>
      )}

      <div>
        <button
          onClick={() => setShowTech((v) => !v)}
          className="text-xs font-semibold underline"
          style={{ color: 'var(--aman-text-dim)' }}
        >
          {t('techDetailsToggle')} {showTech ? '▲' : '▼'}
        </button>
        {showTech && result.metadata && (
          <Card>
            <dl className="grid sm:grid-cols-3 gap-4 text-sm mt-1">
              <div>
                <dt style={{ color: 'var(--aman-text-dim)' }} className="text-xs mb-1">
                  {t('trustScore')}
                </dt>
                <dd className="font-mono font-semibold" style={{ color: 'var(--aman-text)' }}>
                  {result.metadata.trust_score?.toFixed?.(3) ?? '—'}
                </dd>
              </div>
              <div>
                <dt style={{ color: 'var(--aman-text-dim)' }} className="text-xs mb-1">
                  {t('routerThreshold')}
                </dt>
                <dd className="font-mono font-semibold" style={{ color: 'var(--aman-text)' }}>
                  {result.metadata.router_threshold?.toFixed?.(3) ?? '—'}
                </dd>
              </div>
              <div>
                <dt style={{ color: 'var(--aman-text-dim)' }} className="text-xs mb-1">
                  {t('processingTime')}
                </dt>
                <dd className="font-mono font-semibold" style={{ color: 'var(--aman-text)' }}>
                  {result.metadata.processing_time_seconds?.toFixed?.(1) ?? '—'} {t('seconds')}
                </dd>
              </div>
            </dl>
          </Card>
        )}
      </div>
    </div>
  );
}
