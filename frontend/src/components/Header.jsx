import { useLang } from '../i18n.jsx';

export default function Header({ onOpenTransparency, onLogoClick }) {
  const { t, lang, toggleLang } = useLang();

  return (
    <header
      className="sticky top-0 z-20 border-b backdrop-blur-md"
      style={{ background: 'rgba(10,15,28,0.85)', borderColor: 'var(--aman-border)' }}
    >
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-3">
        <button
          onClick={onLogoClick}
          className="flex items-center gap-2.5 text-start"
        >
          <span
            className="flex items-center justify-center w-9 h-9 rounded-xl text-lg font-bold"
            style={{ background: 'var(--aman-teal-dim)', color: 'var(--aman-teal)' }}
          >
            🛰️
          </span>
          <span className="flex flex-col leading-tight">
            <span className="font-extrabold text-base" style={{ color: 'var(--aman-text)' }}>
              {t('brandName')}
            </span>
            <span className="text-[11px]" style={{ color: 'var(--aman-text-dim)' }}>
              {t('brandSub')}
            </span>
          </span>
        </button>

        <div className="flex items-center gap-2">
          <button
            onClick={onOpenTransparency}
            className="text-sm font-semibold px-3 py-2 rounded-lg border transition-colors hover:brightness-110"
            style={{ borderColor: 'var(--aman-border)', color: 'var(--aman-blue)' }}
          >
            {t('howItWorks')}
          </button>
          <button
            onClick={toggleLang}
            className="text-sm font-semibold px-3 py-2 rounded-lg border transition-colors hover:brightness-110"
            style={{ borderColor: 'var(--aman-border)', color: 'var(--aman-text-dim)' }}
            aria-label="toggle language"
          >
            {t('langToggle')}
          </button>
        </div>
      </div>
    </header>
  );
}
