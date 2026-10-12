import { useState } from 'react';
import { useLang, detectDir } from '../i18n.jsx';
import { demoExamples } from '../data/demoExamples';

export default function LandingScreen({ onAnalyze, initialText = '' }) {
  const { t, lang } = useLang();
  const [text, setText] = useState(initialText);

  const canSubmit = text.trim().length > 0;

  function handleExampleClick(example) {
    setText(example.claim);
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!canSubmit) return;
    onAnalyze(text.trim());
  }

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-10 sm:py-16 aman-fade-up">
      <div className="text-center mb-10">
        <h1
          dir="rtl"
          className="font-ar text-3xl sm:text-4xl font-extrabold mb-2 leading-tight"
          style={{ color: 'var(--aman-text)' }}
        >
          {t('heroTitleAr')}
        </h1>
        <h2 className="text-base sm:text-lg font-medium mb-5" style={{ color: 'var(--aman-text-dim)' }}>
          {t('heroTitleEn')}
        </h2>
        <p
          dir={lang === 'ar' ? 'rtl' : 'ltr'}
          className="max-w-xl mx-auto text-sm sm:text-[15px] leading-relaxed"
          style={{ color: 'var(--aman-text-dim)' }}
        >
          {t('heroBody')}
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-3">
        <label className="block text-sm font-semibold" style={{ color: 'var(--aman-text-dim)' }}>
          {t('inputLabel')}
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          dir={detectDir(text)}
          placeholder={t('inputPlaceholder')}
          rows={6}
          className="w-full rounded-2xl border p-4 text-base leading-relaxed resize-none outline-none transition-shadow focus:shadow-lg"
          style={{
            background: 'var(--aman-bg-card)',
            borderColor: 'var(--aman-border)',
            color: 'var(--aman-text)',
          }}
        />

        {!text && (
          <p className="text-xs" style={{ color: 'var(--aman-text-dim)' }}>
            {t('emptyHint')}
          </p>
        )}

        <div className="flex items-center gap-3 pt-1">
          <button
            type="submit"
            disabled={!canSubmit}
            className="px-6 py-3 rounded-xl font-bold text-base transition-transform disabled:opacity-40 disabled:cursor-not-allowed enabled:hover:scale-[1.02]"
            style={{ background: 'var(--aman-teal)', color: '#06211d' }}
          >
            {t('analyzeBtn')}
          </button>
          {text && (
            <button
              type="button"
              onClick={() => setText('')}
              className="px-4 py-3 rounded-xl font-semibold text-sm border"
              style={{ borderColor: 'var(--aman-border)', color: 'var(--aman-text-dim)' }}
            >
              {t('clearBtn')}
            </button>
          )}
        </div>
      </form>

      <div className="mt-12">
        <h3 className="text-sm font-bold mb-1" style={{ color: 'var(--aman-text)' }}>
          {t('examplesTitle')}
        </h3>
        <p className="text-xs mb-4" style={{ color: 'var(--aman-text-dim)' }}>
          {t('examplesSub')}
        </p>
        <div className="grid sm:grid-cols-3 gap-3">
          {demoExamples.map((example) => (
            <button
              key={example.id}
              type="button"
              onClick={() => handleExampleClick(example)}
              className="text-start rounded-xl border p-4 transition-colors hover:brightness-110"
              style={{ background: 'var(--aman-bg-card)', borderColor: 'var(--aman-border)' }}
            >
              <span
                className="inline-block text-[11px] font-bold px-2 py-0.5 rounded-full mb-2"
                style={{ background: 'var(--aman-blue-dim)', color: 'var(--aman-blue)' }}
              >
                {example.tag[lang]}
              </span>
              <p dir={detectDir(example.claim)} className="text-sm leading-relaxed mb-2" style={{ color: 'var(--aman-text)' }}>
                {example.claim}
              </p>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--aman-text-dim)' }}>
                {example.description[lang]}
              </p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
