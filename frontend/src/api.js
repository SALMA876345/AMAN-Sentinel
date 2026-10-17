import { mockVerify, mockHealth } from './mockData';

// Safety rule: this app must NEVER silently show fabricated verification
// results as if they came from AMAN Sentinel. The real backend is used by
// default always. If VITE_API_URL is unset, we fall back to the standard
// local backend address (http://127.0.0.1:8000) rather than to mock data —
// set VITE_API_URL explicitly (e.g. frontend/.env.local) only when the
// backend lives somewhere else.
const API_BASE = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');

// Dev fixtures exist only for UI development with no backend running, and
// can NEVER activate automatically or in a production build. Both of the
// following must be true: Vite's own dev-mode flag (false in `vite build`
// output, so a judge-facing production build can never satisfy this no
// matter what env vars are set) AND an explicit opt-in env var.
const DEV_FIXTURE_ENABLED =
  import.meta.env.DEV && import.meta.env.VITE_ENABLE_DEV_FIXTURE === 'true';

export function isUsingMockBackend() {
  return DEV_FIXTURE_ENABLED;
}

export async function verifyClaim(text, evidenceText = '') {
  if (DEV_FIXTURE_ENABLED) {
    // eslint-disable-next-line no-console
    console.warn(
      '[AMAN] VITE_ENABLE_DEV_FIXTURE=true in dev mode — using the local DEV FIXTURE (src/mockData.js), not a real backend response. This path is unreachable in a production build.'
    );
    return mockVerify(text, evidenceText);
  }
  let res;
  try {
    res = await fetch(`${API_BASE}/api/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, evidence_text: evidenceText }),
    });
  } catch {
    throw new Error(
      `Cannot reach the AMAN Sentinel backend at ${API_BASE}. Make sure the FastAPI server is running and VITE_API_URL points to it.`
    );
  }
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
  if (DEV_FIXTURE_ENABLED) return mockHealth();
  let res;
  try {
    res = await fetch(`${API_BASE}/health`);
  } catch {
    throw new Error(`Cannot reach the AMAN Sentinel backend at ${API_BASE}.`);
  }
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}
