import { useLang } from '../i18n.jsx';

const COLORS = {
  amber: { bg: 'var(--aman-amber-dim)', fg: 'var(--aman-amber)' },
  teal: { bg: 'var(--aman-teal-dim)', fg: 'var(--aman-teal)' },
  green: { bg: 'var(--aman-green-dim)', fg: 'var(--aman-green)' },
  red: { bg: 'var(--aman-red-dim)', fg: 'var(--aman-red)' },
  purple: { bg: 'var(--aman-purple-dim)', fg: 'var(--aman-purple)' },
};

// Determines the two-tier presentation required by the product brief:
// classifier-only results (VERIFIED_AUTOMATICALLY) must never read as a
// factual verdict ("LIKELY RELIABLE/MISLEADING"), while evidence-grounded
// results (VERIFIED_WITH_EVIDENCE / UNCERTAIN) show the evidence_verdict
// directly and use a visually distinct badge style/icon/color.
export function classifyResult(result) {
  const isEvidenceGrounded =
    result.status === 'VERIFIED_WITH_EVIDENCE' || result.status === 'UNCERTAIN';

  if (!isEvidenceGrounded) {
    const isFake = result.classification === 'Fake';
    return {
      tier: 'classifier-only',
      icon: '⚡',
      colorKey: isFake ? 'amber' : 'teal',
      labelKey: isFake ? 'likelyMisleading' : 'likelyReliable',
    };
  }

  let colorKey = 'purple';
  let labelKey = 'verdictInsufficient';
  if (result.evidence_verdict === 'SUPPORTED') {
    colorKey = 'green';
    labelKey = 'verdictSupported';
  } else if (result.evidence_verdict === 'CONTRADICTED') {
    colorKey = 'red';
    labelKey = 'verdictContradicted';
  }
  return { tier: 'evidence-grounded', icon: '🛡', colorKey, labelKey };
}

export default function StatusBadge({ result, size = 'lg' }) {
  const { t } = useLang();
  const info = classifyResult(result);
  const palette = COLORS[info.colorKey];
  const big = size === 'lg';

  return (
    <div
      className={`inline-flex items-center gap-3 rounded-2xl border ${big ? 'px-5 py-4' : 'px-3 py-2'}`}
      style={{ background: palette.bg, borderColor: palette.fg, color: palette.fg }}
    >
      <span className={big ? 'text-3xl' : 'text-xl'}>{info.icon}</span>
      <div className="flex flex-col">
        <span className={`font-extrabold tracking-wide ${big ? 'text-xl' : 'text-sm'}`}>
          {t(info.labelKey)}
        </span>
        <span className="text-xs font-medium opacity-80">
          {t(info.tier === 'classifier-only' ? 'tierClassifierOnly' : 'tierEvidenceGrounded')}
        </span>
      </div>
    </div>
  );
}
