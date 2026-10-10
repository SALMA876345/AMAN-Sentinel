"""Full Google News batchexecute decode test on one URL."""
import json, requests, re
import pandas as pd
from bs4 import BeautifulSoup

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
te = pd.read_excel("data/uae_test_2class.xlsx", sheet_name="Tabelle1")
url = [str(u) for u in te["evidence_url"].tolist() if str(u).startswith("http")][0]

s = requests.Session()
r = s.get(url, headers=H, timeout=20)
soup = BeautifulSoup(r.text, "html.parser")
div = soup.select_one("c-wiz > div")
data_p = div.get("data-p")
print("got data-p:", bool(data_p))
obj = json.loads(data_p.replace("%.@.", '["garturlreq",'))
payload = {"f.req": json.dumps([[["Fbv4je", json.dumps(obj[:-6] + obj[-2:]), "null", "generic"]]])}
hh = {"content-type": "application/x-www-form-urlencoded;charset=UTF-8", **H}
resp = s.post("https://news.google.com/_/DotsSplashUi/data/batchexecute",
              headers=hh, data=payload, timeout=20)
txt = resp.text.replace(")]}'", "").strip()
arr = json.loads(txt.split("\n")[1] if "\n" in txt else txt)
real = json.loads(arr[0][2])[1]
print("DECODED real URL:", real)

# try fetching the real article
ar = s.get(real, headers=H, timeout=20, allow_redirects=True)
asoup = BeautifulSoup(ar.text, "html.parser")
for t in asoup(["script", "style", "nav", "header", "footer"]):
    t.decompose()
body = re.sub(r"\s+", " ", asoup.get_text(" ")).strip()
print("article status:", ar.status_code, "text len:", len(body))
print("sample:", body[:200])
