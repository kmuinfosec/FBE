import csv
import datetime
import gzip
import io
import json
import os
import random
import zipfile
import collections
from multiprocessing import Pool
import numpy as np
from utils.config import strs, path, data_file

def strip_handshake(v):
    i = 0
    while i < len(v) and v[i] == 0:
        i += 1
    return list(v[i:])


def ch_count(v, response_window):
    j = 0
    while j < len(v) and v[j] < 0:
        j += 1
    if j == 0 or not any(x > 0 for x in v[j:j + response_window]):
        return 0
    return j


def fix_width(v, width):
    v = list(v[:width])
    return v + [0] * (width - len(v))


def compress(X):
    order = np.argsort(X == 0, axis=1, kind="stable")
    return np.take_along_axis(X, order, 1)


def head_runs(P, j):
    neg, pos = P < 0, P > 0
    c0 = np.argmin(np.concatenate([neg, np.zeros((len(P), 1), bool)], 1), 1)
    idx = np.arange(P.shape[1])[None, :]
    first_nonpos = np.where((idx >= c0[:, None]) & ~pos, idx, P.shape[1]).min(1)
    c1 = first_nonpos - c0
    h = np.asarray(j) > 0
    return np.where(h, c0, 0), np.where(h & (c0 > 0), c1, 0)


def packet_view(X, j, chunk=200000):
    N = len(X)
    P = np.zeros(X.shape, np.int32)
    c0 = np.zeros(N, np.int16)
    c1 = np.zeros(N, np.int16)
    for s in range(0, N, chunk):
        x = np.asarray(X[s:s + chunk])
        p = compress(x)
        P[s:s + chunk] = p
        a, b = head_runs(p, j[s:s + chunk])
        c0[s:s + chunk] = a
        c1[s:s + chunk] = b
    return P, c0, c1


csv.field_size_limit(1 << 30)

def ppi_to_lengths(ppi):
    d, sz = json.loads(ppi)[1:3]
    return [-int(x) * int(y) for x, y in zip(d, sz)]


def reservoir(rows, key_col, value_cols, size, seed):
    rnd = random.Random(seed)
    res, cnt = {}, collections.Counter()
    for row in rows:
        a = row[key_col]
        c = cnt[a]
        cnt[a] = c + 1
        b = res.setdefault(a, [])
        item = tuple(row[i] for i in value_cols)
        if c < size:
            b.append(item)
        else:
            k = rnd.randrange(c + 1)
            if k < size:
                b[k] = item
    return dict(cnt), res


def scan_tls22(args):
    zpath, member, size = args
    with zipfile.ZipFile(zpath) as z:
        f = io.TextIOWrapper(gzip.GzipFile(fileobj=z.open(member)), newline="")
        r = csv.reader(f)
        h = next(r)
        cnt, res = reservoir(r, h.index("APP"), [h.index("PPI")], size, member)
    print(member, sum(cnt.values()), flush=True)
    return member, cnt, res


def cesnet_tls22(cfg):
    sec, rw = cfg["cesnet_tls22"], cfg.getint("flows", "response_window")
    zpath = path(cfg, "cesnet_tls22_zip")
    members = sorted(m for m in zipfile.ZipFile(zpath).namelist() if m.endswith(".csv.gz"))
    weeks = sorted({m.split("/")[0] for m in members})
    tr_weeks = set(weeks[:len(weeks) // 2])
    with Pool(min(len(members), sec.getint("workers"))) as p:
        out = p.map(scan_tls22, [(zpath, m, sec.getint("reservoir")) for m in members])
    pool = {0: {}, 2: {}}
    for m, cnt, res in out:
        s = 0 if m.split("/")[0] in tr_weeks else 2
        for a, b in res.items():
            pool[s].setdefault(a, []).extend(x[0] for x in b)
    rng = np.random.default_rng(0)
    rows = {0: {}, 2: {}}
    for s in (0, 2):
        for a, b in pool[s].items():
            vs = [v for v in map(ppi_to_lengths, b) if ch_count(v, rw)]
            if len(vs) > sec.getint("cap"):
                vs = [vs[i] for i in rng.choice(len(vs), sec.getint("cap"), replace=False)]
            rows[s][a] = vs
    keep = sorted(a for a in rows[0] if len(rows[0][a]) >= sec.getint("min_train") and len(rows[2].get(a, [])) >= sec.getint("min_test"))
    width = cfg.getint("flows", "width")
    X, y, sp = [], [], []
    for s in (0, 2):
        for k, a in enumerate(keep):
            for v in rows[s][a]:
                X.append(fix_width(v, width))
                y.append(k)
                sp.append(s)
    return dict(X=X, y=y, split=sp, kind=np.zeros(len(y), np.int8), services=keep)


def cipherspectrum(cfg):
    import pyarrow.parquet as pq
    raw, width, rw = path(cfg, "cipherspectrum_dir"), cfg.getint("flows", "width"), cfg.getint("flows", "response_window")
    cut = datetime.datetime.fromisoformat(cfg.get("cipherspectrum", "time_cut")).replace(tzinfo=datetime.timezone.utc).timestamp()
    X, y, names, pcaps, times, chromium = [], [], {}, [], [], []
    for f in ("train", "valid", "test"):
        t = pq.read_table(f"{raw}/{f}_flow.parquet", columns=["From", "SPLT-Data", "Label", "Timestamp"]).to_pydict()
        for fr, v, lab, ts in zip(t["From"], t["SPLT-Data"], t["Label"], t["Timestamp"]):
            names.setdefault(lab, set()).add(fr.split("/")[0])
            X.append(fix_width(strip_handshake(v), width))
            y.append(lab)
            pcaps.append(fr.split(".pcap")[0])
            times.append(ts)
            chromium.append(int("_chromium_" in fr))
    assert all(len(v) == 1 for v in names.values())
    services = [next(iter(names[k])) for k in range(len(names))]
    first = collections.defaultdict(lambda: np.inf)
    for p, ts in zip(pcaps, times):
        first[p] = min(first[p], ts)
    split = np.array([0 if first[p] < cut else 2 for p in pcaps])
    y, chromium = np.asarray(y), np.asarray(chromium, np.int8)
    sec = cfg["cipherspectrum"]
    use = chromium == {"firefox": 0, "chromium": 1}[sec.get("browser")] if sec.get("browser", "") in ("firefox", "chromium") else np.ones(len(y), bool)
    tr = np.bincount(y[use & (split == 0)], minlength=len(services))
    te = np.bincount(y[use & (split == 2)], minlength=len(services))
    keep = [c for c in range(len(services)) if tr[c] >= sec.getint("min_train") and te[c] >= sec.getint("min_test")]
    remap = {c: k for k, c in enumerate(keep)}
    sel = np.flatnonzero(use & np.isin(y, keep))
    services = [services[c] for c in keep]
    return dict(X=[X[i] for i in sel], y=[remap[c] for c in y[sel]], split=split[sel], kind=np.zeros(len(sel), np.int8), services=services)


def kmu_tls(cfg):
    sec = cfg["kmu_tls"]
    r = np.load(path(cfg, "kmu_tls_flows"), allow_pickle=True)
    X, y, split, ver, services = r["X"], r["y"].astype(np.int64), r["split"], r["tls_version"], r["services"].tolist()
    tls13 = ver == 3
    ctr = np.bincount(y[tls13 & (split == 0)], minlength=len(services))
    cte = np.bincount(y[tls13 & (split == 2)], minlength=len(services))
    keep = [c for c in range(len(services)) if ctr[c] >= sec.getint("min_train") and cte[c] >= sec.getint("min_test")]
    remap = {c: k for k, c in enumerate(keep)}
    sel = np.flatnonzero(tls13 & np.isin(y, keep))
    return dict(X=X[sel].astype(np.int32), y=[remap[c] for c in y[sel]], split=split[sel], kind=np.zeros(len(sel), np.int8),
                services=[services[c] for c in keep])


def echpair(cfg):
    import pyarrow.parquet as pq
    width = cfg.getint("flows", "width")
    hosts, labels = strs(cfg, "echpair", "hosts"), strs(cfg, "echpair", "probe_labels")
    H = pq.read_table(path(cfg, "echpair_hosts_parquet"), columns=["kind", "y", "host", "month", "pcap", "v"]).to_pydict()
    B = pq.read_table(path(cfg, "echpair_probe_parquet"), columns=["kind", "svc", "y", "host", "host_group", "month", "t0", "pcap", "v"]).to_pydict()
    X, y, kind, month, pcap, t0, group, source = [], [], [], [], [], [], [], []
    for r in zip(H["kind"], H["y"], H["host"], H["month"], H["pcap"], H["v"]):
        X.append(fix_width(strip_handshake(r[5]), width))
        y.append(r[1])
        kind.append(int(r[0] == "ech"))
        month.append(r[3])
        pcap.append(r[4])
        t0.append(0.0)
        group.append("")
        source.append(0)
    for r in zip(B["kind"], B["svc"], B["y"], B["host"], B["host_group"], B["month"], B["t0"], B["pcap"], B["v"]):
        name = r[1].replace("probe_", "")
        X.append(fix_width(strip_handshake(r[8]), width))
        y.append(len(hosts) + labels.index(name) if name in labels else -1)
        kind.append(int(r[0] == "ech"))
        month.append(r[5])
        pcap.append(r[7])
        t0.append(float(r[6]))
        group.append(r[4])
        source.append(1)
    services = hosts + labels
    return dict(X=X, y=y, split=np.full(len(y), 2, np.int8), kind=np.asarray(kind, np.int8), services=services, month=np.array(month), pcap=np.array(pcap),
                t0=np.asarray(t0, np.float64), host_group=np.array(group), source=np.asarray(source, np.int8))


BUILDERS = {"cesnet_tls22": cesnet_tls22, "cipherspectrum": cipherspectrum, "kmu_tls": kmu_tls, "echpair": echpair}


def finalize(cfg, name, d):
    rw = cfg.getint("flows", "response_window")
    X = np.asarray(d.pop("X"), np.int32)
    j = np.array([ch_count(list(x), rw) for x in X], np.int8)
    kind = np.asarray(d["kind"], np.int8)
    P, c0, c1 = packet_view(X, j.astype(np.int64))
    assert np.abs(P).max() < 32768
    out = {k: np.asarray(v) for k, v in d.items()}
    out.update(P=P.astype(np.int16), c0=c0, c1=c1, kind=kind, y=np.asarray(d["y"], np.int32))
    os.makedirs(path(cfg, "data_dir"), exist_ok=True)
    np.savez_compressed(data_file(cfg, name), **out)
    h = j > 0
    print(name, "flows", len(X), "classes", len(out["services"]), "with_ch", int(h.sum()),
          "c0_dist", dict(zip(*map(lambda a: a.tolist(), np.unique(c0[h], return_counts=True)))), flush=True)


def run(cfg, name):
    finalize(cfg, name, BUILDERS[name](cfg))
