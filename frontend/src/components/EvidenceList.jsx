import { useLang, detectDir } from '../i18n.jsx';

export default function EvidenceList({ evidence }) {
  const { t } = useLang();
  if (!evidence || evidence.length === 0) {
    return (
      <p className="text-sm italic" style={{ color: 'var(--aman-text-dim)' }}>
        {t('noEvidence')}
      </p>
    );
  }

  return (
    <ul className="space-y-3">
      {evidence.map((item, idx) => (
        <li
          key={item.id || idx}
          className="rounded-xl border p-4"
          style={{ background: 'var(--aman-bg-raised)', borderColor: 'var(--aman-border)' }}
        >
          <p
            dir={detectDir(item.text)}
            className="text-sm leading-relaxed mb-3"
            style={{ color: 'var(--aman-text)' }}
          >
            {item.text}
          </p>
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs" style={{ color: 'var(--aman-text-dim)' }}>
            {item.source && (
              <span>
                <strong style={{ color: 'var(--aman-blue)' }}>{t('evidenceSource')}:</strong>{' '}
                {item.source}
              </span>
            )}
            {item.retrieval_method && (
              <span>
                <strong style={{ color: 'var(--aman-blue)' }}>{t('evidenceMethod')}:</strong>{' '}
                {item.retrieval_method}
              </span>
            )}
            {typeof item.retrieval_score === 'number' && (
              <span>
                <strong style={{ color: 'var(--aman-blue)' }}>{t('evidenceScore')}:</strong>{' '}
                {item.retrieval_score.toFixed(3)}
              </span>
            )}
          </div>
        </li>
      ))}
    </ul>
  );
}
