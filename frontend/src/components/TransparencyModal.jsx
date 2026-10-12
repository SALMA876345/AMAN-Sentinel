import { useEffect } from 'react';
import { useLang } from '../i18n.jsx';

function Row({ label, value }) {
  return (
    <div className="flex items-baseline justify-between py-1.5 border-b last:border-b-0" style={{ borderColor: 'var(--aman-border)' }}>
      <span className="text-sm" style={{ color: 'var(--aman-text-dim)' }}>{label}</span>
      <span className="text-sm font-mono font-bold" style={{ color: 'var(--aman-text)' }}>{value}</span>
    </div>
  );
}

export default function TransparencyModal({ onClose }) {
  const { t, lang, dir } = useLang();

  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)' }}
      onClick={onClose}
    >
      <div
        dir={dir}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-2xl max-h-[85vh] overflow-y-auto aman-scroll rounded-2xl border p-6"
        style={{ background: 'var(--aman-bg-raised)', borderColor: 'var(--aman-border)' }}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-extrabold" style={{ color: 'var(--aman-text)' }}>
            {t('modalTitle')}
          </h2>
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg border text-sm font-semibold"
            style={{ borderColor: 'var(--aman-border)', color: 'var(--aman-text-dim)' }}
          >
            {t('modalClose')} ✕
          </button>
        </div>

        <p className="text-sm leading-relaxed mb-6" style={{ color: 'var(--aman-text)' }}>
          {t('modalIntro')}
        </p>

        <section className="mb-6">
          <h3 className="text-sm font-bold mb-2" style={{ color: 'var(--aman-teal)' }}>
            {t('modalClassifierHeading')}
          </h3>
          <div className="rounded-xl border p-3" style={{ borderColor: 'var(--aman-border)' }}>
            <Row label={lang === 'ar' ? 'الدقة على بيانات الاختبار' : 'Test accuracy'} value="86.45%" />
            <Row label={lang === 'ar' ? 'المتوسط التوافقي الكُلّي (Macro F1)' : 'Macro F1'} value="0.8640" />
            <Row label="MCC" value="0.7281" />
          </div>
        </section>

        <section className="mb-6">
          <h3 className="text-sm font-bold mb-2" style={{ color: 'var(--aman-amber)' }}>
            {t('modalAgreementHeading')}
          </h3>
          <div className="rounded-xl border p-3 mb-2" style={{ borderColor: 'var(--aman-border)' }}>
            <Row label={lang === 'ar' ? 'اتفاق النماذج الثلاثة (67% من الحالات)' : 'All 3 models agree (67% of cases)'} value="92.6%" />
            <Row label={lang === 'ar' ? 'انقسام 2 إلى 1 (33% من الحالات)' : 'Split 2-to-1 (33% of cases)'} value="64.9%" />
          </div>
          <p className="text-xs leading-relaxed" style={{ color: 'var(--aman-text-dim)' }}>
            {t('modalAgreementBody')}
          </p>
        </section>

        <section className="mb-6">
          <h3 className="text-sm font-bold mb-2" style={{ color: 'var(--aman-blue)' }}>
            {t('modalSelectiveHeading')}
          </h3>
          <div className="rounded-xl border p-3 mb-2" style={{ borderColor: 'var(--aman-border)' }}>
            <Row label={lang === 'ar' ? 'التغطية' : 'Coverage'} value="76.90%" />
            <Row label={lang === 'ar' ? 'الدقة الانتقائية' : 'Selective accuracy'} value="92.65%" />
          </div>
          <p className="text-xs leading-relaxed" style={{ color: 'var(--aman-text-dim)' }}>
            {t('modalSelectiveBody')}
          </p>
        </section>

        <section className="mb-2">
          <h3 className="text-sm font-bold mb-2" style={{ color: 'var(--aman-purple)' }}>
            {t('modalRetrievalHeading')}
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs mb-3" style={{ color: 'var(--aman-text)' }}>
              <thead>
                <tr style={{ color: 'var(--aman-text-dim)' }}>
                  <th className="text-start py-1.5">{lang === 'ar' ? 'الطريقة' : 'Method'}</th>
                  <th className="text-start py-1.5">Recall@5</th>
                  <th className="text-start py-1.5">Recall@10</th>
                  <th className="text-start py-1.5">MRR</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t" style={{ borderColor: 'var(--aman-border)' }}>
                  <td className="py-1.5">BM25</td>
                  <td className="font-mono">0.076</td>
                  <td className="font-mono">0.140</td>
                  <td className="font-mono">0.0618</td>
                </tr>
                <tr className="border-t" style={{ borderColor: 'var(--aman-border)' }}>
                  <td className="py-1.5">Dense</td>
                  <td className="font-mono">0.092</td>
                  <td className="font-mono">0.144</td>
                  <td className="font-mono">0.0596</td>
                </tr>
                <tr className="border-t" style={{ borderColor: 'var(--aman-border)' }}>
                  <td className="py-1.5 font-semibold" style={{ color: 'var(--aman-teal)' }}>
                    Hybrid-RRF {lang === 'ar' ? '(المختارة)' : '(selected)'}
                  </td>
                  <td className="font-mono font-semibold" style={{ color: 'var(--aman-teal)' }}>0.108</td>
                  <td className="font-mono font-semibold" style={{ color: 'var(--aman-teal)' }}>0.202</td>
                  <td className="font-mono font-semibold" style={{ color: 'var(--aman-teal)' }}>0.0787</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div
            className="rounded-xl border p-3 text-xs leading-relaxed"
            style={{ borderColor: 'var(--aman-amber)', color: 'var(--aman-amber)', background: 'var(--aman-amber-dim)' }}
          >
            ⚠️ {t('modalRetrievalDisclosure')}
          </div>
        </section>
      </div>
    </div>
  );
}
