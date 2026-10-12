import { useLang } from '../i18n.jsx';

const LEVEL_KEY = {
  FAMILIAR: 'noveltyFamiliar',
  'MODERATELY NOVEL': 'noveltyModerate',
  'HIGHLY NOVEL': 'noveltyHigh',
};

const LEVEL_COLOR = {
  FAMILIAR: 'var(--aman-teal)',
  'MODERATELY NOVEL': 'var(--aman-amber)',
  'HIGHLY NOVEL': 'var(--aman-purple)',
};

export default function NoveltyBadge({ novelty }) {
  const { t } = useLang();
  if (!novelty) return null;
  const level = novelty.level || 'FAMILIAR';
  const pct = Math.round(Math.max(0, Math.min(1, novelty.score ?? 0)) * 100);
  const color = LEVEL_COLOR[level] || 'var(--aman-text-dim)';

  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-sm font-semibold" style={{ color: 'var(--aman-text-dim)' }}>
          {t('noveltyLabel')}
        </span>
        <span className="text-sm font-bold" style={{ color }}>
          {t(LEVEL_KEY[level] || 'noveltyFamiliar')} · {pct}%
        </span>
      </div>
      <div className="h-2.5 w-full rounded-full overflow-hidden" style={{ background: 'var(--aman-border)' }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <p className="text-xs mt-2 leading-relaxed" style={{ color: 'var(--aman-text-dim)' }}>
        {t('noveltyExplain')}
      </p>
    </div>
  );
}
