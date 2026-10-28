"""Run a Fluster HEVC test-suite definition against rust_h265's dump_frames.

Usage: python3 -I run_suite.py SUITE.json DECODER WORKDIR [--only NAME ...]
Writes WORKDIR/results-<suite>.json and prints a summary.
"""
import concurrent.futures as cf, hashlib, json, os, subprocess, sys, urllib.request, zipfile

def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(tv, zips):
    dst = os.path.join(zips, tv["name"] + ".zip")
    if not (os.path.exists(dst) and md5_file(dst) == tv["source_checksum"]):
        req = urllib.request.Request(tv["source"], headers={"User-Agent": "rust_h265-conformance"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dst + ".part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(dst + ".part", dst)
    got = md5_file(dst)
    if got != tv["source_checksum"]:
        raise RuntimeError(f"zip checksum mismatch {got}")
    return dst

def extract(tv, zpath, vecdir):
    want = os.path.basename(tv["input_file"])
    os.makedirs(vecdir, exist_ok=True)
    out = os.path.join(vecdir, want)  # fixed, safe name; never use member paths
    if os.path.exists(out):
        return out
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if os.path.basename(n) == want]
        if not names:
            raise RuntimeError(f"{want} not in zip: {z.namelist()[:6]}")
        with z.open(names[0]) as src, open(out + ".part", "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    os.replace(out + ".part", out)
    return out

def run_one(tv, decoder, work):
    name = tv["name"]
    res = {"name": name, "expected": tv["result"], "output_format": tv.get("output_format")}
    try:
        z = fetch(tv, os.path.join(work, "zips"))
        bitstream = extract(tv, z, os.path.join(work, "vectors", name))
    except Exception as e:
        res.update(status="fetch_error", detail=str(e)[:300]); return res
    out = os.path.join(work, "out", name + ".yuv")
    try:
        p = subprocess.run([decoder, bitstream, out], capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        res.update(status="timeout"); return res
    err = (p.stderr or "").strip().splitlines()
    res["stderr_tail"] = err[-3:]
    if p.returncode != 0:
        panicked = any("panicked" in l for l in err)
        res.update(status="panic" if panicked else "decode_error",
                   detail=next((l for l in err if "panicked" in l or "decode error" in l), err[-1] if err else ""))
    elif not os.path.exists(out):
        res.update(status="no_output")
    else:
        got = md5_file(out)
        res.update(md5=got, status="pass" if got == tv["result"] else "mismatch")
    if res["status"] == "pass" and os.path.exists(out):
        os.remove(out)  # keep failures for analysis, drop passes to save space
    return res

def main():
    suite, decoder, work = sys.argv[1], sys.argv[2], sys.argv[3]
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
    for d in ("zips", "vectors", "out"):
        os.makedirs(os.path.join(work, d), exist_ok=True)
    spec = json.load(open(suite))
    tvs = [t for t in spec["test_vectors"] if not only or t["name"] in only]
    with cf.ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda t: run_one(t, decoder, work), tvs))
    path = os.path.join(work, f"results-{spec['name']}.json")
    json.dump(results, open(path, "w"), indent=1)
    from collections import Counter
    c = Counter(r["status"] for r in results)
    print(f"{spec['name']}: {c.get('pass',0)}/{len(results)} pass  {dict(c)}")
    for r in results:
        if r["status"] != "pass":
            print(f"  {r['status']:12} {r['name']:40} {r.get('detail','')[:110]}")
    print("results:", path)

if __name__ == "__main__":
    main()
