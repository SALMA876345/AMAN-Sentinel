import { useLang } from '../i18n.jsx';

export default function ConfidenceMeter({ confidence }) {
  const { t } = useLang();
  const pct = Math.round(Math.max(0, Math.min(1, confidence ?? 0)) * 100);
  const color = pct >= 80 ? 'var(--aman-teal)' : pct >= 55 ? 'var(--aman-amber)' : 'var(--aman-red)';

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-sm font-semibold" style={{ color: 'var(--aman-text-dim)' }}>
          {t('confidenceLabel')}
        </span>
        <span className="text-lg font-bold" style={{ color }}>
          {pct}%
        </span>
      </div>
      <div className="h-2.5 w-full rounded-full overflow-hidden" style={{ background: 'var(--aman-border)' }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}
