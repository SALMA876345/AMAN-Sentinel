import { useCallback, useEffect, useState } from 'react';
import { useLang } from './i18n.jsx';
import { verifyClaim } from './api.js';
import Header from './components/Header.jsx';
import LandingScreen from './components/LandingScreen.jsx';
import AnalysisProgress from './components/AnalysisProgress.jsx';
import ResultDashboard from './components/ResultDashboard.jsx';
import TransparencyModal from './components/TransparencyModal.jsx';

// Screens: 'landing' -> 'analyzing' -> 'result' (or 'error' on failure)
export default function App() {
  const { t, dir } = useLang();
  const [screen, setScreen] = useState('landing');
  const [inputText, setInputText] = useState('');
  const [startTime, setStartTime] = useState(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [finalStages, setFinalStages] = useState(null);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [showTransparency, setShowTransparency] = useState(false);

  // Elapsed-time ticker, only while a request is in flight.
  useEffect(() => {
    if (screen !== 'analyzing' || !startTime) return;
    const id = setInterval(() => setElapsedMs(Date.now() - startTime), 200);
    return () => clearInterval(id);
  }, [screen, startTime]);

  const handleAnalyze = useCallback(async (text) => {
    setInputText(text);
    setResult(null);
    setFinalStages(null);
    setErrorMsg('');
    setElapsedMs(0);
    setStartTime(Date.now());
    setScreen('analyzing');

    try {
      const res = await verifyClaim(text);
      setResult(res);
      setFinalStages(Array.isArray(res.processing_stages) ? res.processing_stages : []);
    } catch (err) {
      setErrorMsg(err?.message || 'Unknown error');
      setScreen('error');
    }
  }, []);

  const handleProgressContinue = useCallback(() => {
    setScreen('result');
  }, []);

  const handleReset = useCallback(() => {
    setScreen('landing');
    setResult(null);
    setFinalStages(null);
    setErrorMsg('');
  }, []);

  const handleRetry = useCallback(() => {
    if (inputText) handleAnalyze(inputText);
    else setScreen('landing');
  }, [inputText, handleAnalyze]);

  return (
    <div dir={dir} className="min-h-screen bg-radial-aman">
      <Header
        onOpenTransparency={() => setShowTransparency(true)}
        onLogoClick={handleReset}
      />

      {screen === 'landing' && (
        <LandingScreen onAnalyze={handleAnalyze} initialText={inputText} />
      )}

      {screen === 'analyzing' && (
        <AnalysisProgress
          elapsedMs={elapsedMs}
          finalStages={finalStages}
          onContinue={handleProgressContinue}
        />
      )}

      {screen === 'result' && result && (
        <ResultDashboard result={result} inputText={inputText} onReset={handleReset} />
      )}

      {screen === 'error' && (
        <div className="max-w-md mx-auto px-4 py-20 text-center aman-fade-up">
          <div
            className="inline-flex items-center justify-center w-14 h-14 rounded-2xl mb-4 text-2xl"
            style={{ background: 'var(--aman-red-dim)' }}
          >
            ⚠️
          </div>
          <h1 className="text-lg font-extrabold mb-2" style={{ color: 'var(--aman-text)' }}>
            {t('errorTitle')}
          </h1>
          <p className="text-sm mb-6 break-words" style={{ color: 'var(--aman-text-dim)' }}>
            {errorMsg}
          </p>
          <button
            onClick={handleRetry}
            className="px-5 py-2.5 rounded-xl font-bold text-sm"
            style={{ background: 'var(--aman-teal)', color: '#06211d' }}
          >
            {t('retryBtn')}
          </button>
        </div>
      )}

      {showTransparency && <TransparencyModal onClose={() => setShowTransparency(false)} />}
    </div>
  );
}
