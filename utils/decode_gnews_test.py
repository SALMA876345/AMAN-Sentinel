"""Test if we can decode a Google News RSS link to the real article URL."""
import base64, json, requests, re
import pandas as pd

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
te = pd.read_excel("data/uae_test_2class.xlsx", sheet_name="Tabelle1")
url = [str(u) for u in te["evidence_url"].tolist() if str(u).startswith("http")][0]
print("URL:", url[:90])

# Attempt 1: naive base64 decode of the token
tok = url.split("/articles/")[1].split("?")[0]
try:
    pad = tok + "=" * (-len(tok) % 4)
    raw = base64.urlsafe_b64decode(pad)
    found = re.findall(rb"https?://[^\s\x00-\x1f\"']+", raw)
    print("base64 decode -> urls found:", [f.decode('utf-8','ignore')[:80] for f in found[:3]] or "NONE")
except Exception as e:
    print("base64 decode failed:", e)

# Attempt 2: Google batchexecute decoder (current method)
try:
    s = requests.Session()
    r = s.get(url, headers=H, timeout=15)
    m = re.search(r'data-n-a-id="([^"]+)"', r.text)
    cw = re.search(r'c-wiz[^>]*data-p="([^"]+)"', r.text)
    print("has c-wiz data-p:", bool(cw), " has data-n-a-id:", bool(m))
    print("page length:", len(r.text))
except Exception as e:
    print("batchexecute probe failed:", type(e).__name__, e)
