import { mockVerify, mockHealth } from './mockData';

// When VITE_API_URL is unset, the app falls back to the local DEV FIXTURE
// in mockData.js (clearly labeled there) purely so the UI can be built and
// demoed before/without the live FastAPI backend. Set VITE_API_URL (e.g. in
// frontend/.env.local as VITE_API_URL=http://localhost:8000) to talk to the
// real pipeline.
const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '');

export function isUsingMockBackend() {
  return !API_BASE;
}

export async function verifyClaim(text, evidenceText = '') {
  if (!API_BASE) {
    // eslint-disable-next-line no-console
    console.warn(
      '[AMAN] VITE_API_URL is not set — using the local DEV FIXTURE (src/mockData.js), not a real backend response.'
    );
    return mockVerify(text, evidenceText);
  }
  const res = await fetch(`${API_BASE}/api/verify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, evidence_text: evidenceText }),
  });
  if (!res.ok) {
    let detail = '';
    try {
      detail = await res.text();
    } catch {
      // ignore
    }
    throw new Error(`AMAN backend returned ${res.status}${detail ? `: ${detail}` : ''}`);
  }
  return res.json();
}

export async function checkHealth() {
  if (!API_BASE) return mockHealth();
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}
