"""Feasibility probe: can we resolve the UAE evidence_url (Google News RSS) to article text?"""
import pandas as pd, requests, re, sys
from bs4 import BeautifulSoup

te = pd.read_excel("data/uae_test_2class.xlsx", sheet_name="Tabelle1")
urls = [str(u) for u in te["evidence_url"].tolist() if str(u).startswith("http")][:6]
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

for i, u in enumerate(urls):
    print(f"\n[{i}] {u[:80]}...")
    try:
        r = requests.get(u, headers=H, timeout=15, allow_redirects=True)
        print(f"  status={r.status_code}  final={r.url[:80]}")
        # google news rss often returns an HTML page with a redirect link/script
        soup = BeautifulSoup(r.text, "html.parser")
        a = soup.find("a", href=True)
        canonical = soup.find("link", rel="canonical")
        print(f"  is_google={'news.google' in r.url}  first_link={(a['href'][:70] if a else None)}")
        print(f"  canonical={(canonical['href'][:70] if canonical else None)}")
        txt = re.sub(r"\s+", " ", soup.get_text(" "))
        print(f"  page_text_len={len(txt)}  sample={txt[:120]}")
    except Exception as e:
        print(f"  ERROR {type(e).__name__}: {e}")
