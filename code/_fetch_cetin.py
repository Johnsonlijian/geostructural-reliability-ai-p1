"""One-off fetcher for the Cetin et al. (2018) PMC Open-Access package (DOI 10.1016/j.dib.2018.08.043)."""
import os
import tarfile
import urllib.request

BASE = r"R:\NAS_DRIVE\IMUT\1-Research_Output\1-Papers\1_In_Preparation\2026-GeoStructural-Reliability-AI\code\data\raw\cetin2018_spt"
TGZ = os.path.join(BASE, "PMC6126194.tar.gz")
URLS = [
    "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/43/4a/PMC6126194.tar.gz",
    "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/43/4a/PMC6126194.tar.gz",
]

ok = False
for url in URLS:
    try:
        if url.startswith("ftp://"):
            with urllib.request.urlopen(url, timeout=180) as r, open(TGZ, "wb") as f:
                f.write(r.read())
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(TGZ, "wb") as f:
                f.write(r.read())
        print("DL OK ", url, os.path.getsize(TGZ), "bytes")
        ok = True
        break
    except Exception as e:
        print("DL FAIL", url, "->", repr(e))

if ok:
    with tarfile.open(TGZ) as t:
        names = t.getnames()
        t.extractall(BASE)
    print("EXTRACTED:")
    for n in names:
        print("  ", n)
else:
    print("ALL DOWNLOADS FAILED")
