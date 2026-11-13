import collections
import json
import os
import time
import numpy as np
from sklearn.metrics import f1_score
from utils.config import strs, ints, mapping, data_file, results_dir
from models import lgbm, cnn

def standard_split(d):
    split, kind = d["split"], d["kind"]
    tr = np.flatnonzero((split == 0) & (kind == 0))
    tests = {"unchanged": np.flatnonzero((split == 2) & (kind == 0))}
    if ((split == 2) & (kind == 1)).any():
        tests["real_ech"] = np.flatnonzero((split == 2) & (kind == 1))
    return tr, tests


def session_sets(rows, y, kind, pcap, seed, per):
    rng = np.random.default_rng(seed)
    pool = collections.defaultdict(list)
    for i in rows:
        pool[(int(y[i]), "ech" if kind[i] else "plain")].append(i)
    S = {"tr": [], "plain": [], "real": []}
    cls = sorted({c for c, _ in pool})
    perm = {c: rng.permutation(pool[(c, "plain")]) for c in cls}
    for c in cls:
        S["tr"] += perm[c][:per].tolist()
    used = {pcap[i] for i in S["tr"]}
    for c in cls:
        rest = [i for i in perm[c][per:] if pcap[i] not in used]
        ec = [i for i in rng.permutation(pool[(c, "ech")]) if pcap[i] not in used]
        S["plain"] += rest[:per]
        S["real"] += ec[:per]
    S = {k: np.sort(np.array(v, dtype=np.int64)) for k, v in S.items()}
    for b in ("plain", "real"):
        assert not {pcap[i] for i in S["tr"]} & {pcap[i] for i in S[b]}
    return S


def echpair_split(d, cfg, seed):
    y, kind, pcap, source, month, t0, group = (d[k] for k in ("y", "kind", "pcap", "source", "month", "t0", "host_group"))
    per = cfg.getint("echpair", "per_class")
    months = set(strs(cfg, "echpair", "host_months"))
    exclude = cfg.get("echpair", "probe_exclude_host_group")
    host_rows = [i for i in np.flatnonzero(source == 0) if month[i] in months]
    best = {}
    for i in np.flatnonzero(source == 1):
        if y[i] < 0 or group[i] == exclude:
            continue
        key = (int(kind[i]), pcap[i])
        if key not in best or t0[i] < t0[best[key]]:
            best[key] = i
    probe_rows = sorted(best.values())
    A = session_sets(host_rows, y, kind, pcap, seed, per)
    B = session_sets(probe_rows, y, kind, pcap, seed, per)
    S = {k: np.concatenate([A[k], B[k]]) for k in A}
    return S["tr"], {"unchanged": S["plain"], "real_ech": S["real"]}


def seg(total, mss):
    q, r = divmod(int(total), int(mss))
    return [mss] * q + ([r] if r else [])


def const_table(P, c0):
    tot = -P.astype(np.int64).cumsum(1)[np.arange(len(P)), np.maximum(c0 - 1, 0)]
    h = c0 > 0
    tab = {int(v): collections.Counter(tot[h & (c0 == v)].tolist()).most_common(1)[0][0] for v in np.unique(c0[h])}
    return tab, collections.Counter(tot[h].tolist()).most_common(1)[0][0]


def transform(P, c0, c1, how, const, cap, mss, pqc_client, pqc_server):
    P, c0, c1 = P.copy(), c0.copy(), c1.copy()
    W = P.shape[1]
    for i in np.flatnonzero(c0 > 0):
        a, b = int(c0[i]), int(c1[i])
        v = P[i]
        ln = int(np.count_nonzero(v))
        ch = -int(v[:a].sum())
        mc = int(-v[:a].min()) if a > 1 else mss
        srv = v[a:a + b].tolist()
        if how == "ch_length_replaced":
            ch = int(const[0].get(a, const[1]))
        elif how == "emulated_pqc":
            ch += pqc_client
        else:
            raise ValueError(how)
        new = [-x for x in seg(ch, mc)]
        if how == "emulated_pqc" and b:
            srv = seg(sum(srv) + pqc_server, max(srv) if b > 1 else mss)
        sequence = (new + srv + v[a + b:ln].tolist())[:min(cap, W)]
        P[i] = 0
        P[i, :len(sequence)] = sequence
        c0[i] = min(len(new), len(sequence))
        c1[i] = max(0, min(len(srv), len(sequence) - len(new)))
    return P, c0, c1


def take(P, off, cnt, w):
    off = np.broadcast_to(np.asarray(off), (len(P),))
    cnt = np.clip(np.broadcast_to(np.asarray(cnt), (len(P),)), 0, w)
    k = np.arange(w)[None, :]
    pos = off[:, None] + k
    ok = (k < cnt[:, None]) & (pos < P.shape[1])
    out = np.take_along_axis(P, np.minimum(pos, P.shape[1] - 1), 1).astype(np.float32)
    out[~ok] = 0
    return out


def build(P, c0, c1, name, n):
    h = c0 > 0
    if name == "ch_included":
        return take(P, 0, n, n)
    if name == "ch_burst_removed":
        return np.where(h[:, None], take(P, c0, n - c0, n), take(P, 0, n, n))
    if name == "fbe":
        return take(P, c0 + c1, n - c0 - c1, n)
    if name.startswith("drop"):
        k = int(name[4:])
        return take(P, k, n - k, n)
    raise ValueError(name)


def scores(yt, yp):
    return {"macro_f1": float(f1_score(yt, yp, average="macro", labels=sorted(set(yt.tolist())), zero_division=0))}


MODELS = {"lgbm": lgbm, "cnn": cnn}


def setup(cfg, dataset, seed):
    d = np.load(data_file(cfg, dataset), allow_pickle=True)
    if dataset == "echpair":
        tr, T = echpair_split(d, cfg, seed)
    else:
        tr, T = standard_split(d)
    tests = {t: (idx, None) for t, idx in T.items()}
    conditions = strs(cfg, "experiment", "conditions")
    for cond, allowed in (("ch_length_replaced", True), ("emulated_pqc", dataset in strs(cfg, "experiment", "pqc_datasets"))):
        if cond in conditions and allowed:
            tests[cond] = (T["unchanged"], cond)
    return dict(P=d["P"], c0=d["c0"].astype(np.int64), c1=d["c1"].astype(np.int64), y=d["y"], tr=tr, tests=tests, nc=len(d["services"]),
                cap=cfg.getint("flows", "ppi_max") if dataset.startswith("cesnet") else d["P"].shape[1])


def test_views(cfg, D):
    P, c0, c1, tr = D["P"], D["c0"], D["c1"], D["tr"]
    const = const_table(P[tr], c0[tr])
    mss, pc, ps = cfg.getint("flows", "mss"), cfg.getint("emulation", "pqc_client_bytes"), cfg.getint("emulation", "pqc_server_bytes")
    V = {}
    for t, (idx, how) in D["tests"].items():
        V[t] = (P[idx], c0[idx], c1[idx]) if how is None else transform(P[idx], c0[idx], c1[idx], how, const, D["cap"], mss, pc, ps)
    return const, V


def budgets(cfg, dataset, D):
    wmax = int(np.count_nonzero(D["P"][D["tr"]], 1).max())
    nmax = min(wmax, mapping(cfg, "experiment", "nmax").get(dataset, wmax))
    return [n for n in ints(cfg, "experiment", "ns") if n <= nmax]


def run(cfg, model, dataset, input_name, seed):
    seed = int(seed)
    M = MODELS[model]
    t0 = time.time()
    D = setup(cfg, dataset, seed)
    P, c0, c1, y, tr = D["P"], D["c0"], D["c1"], D["y"], D["tr"]
    _, V = test_views(cfg, D)
    name = os.path.join(results_dir(cfg, model, dataset), f"{input_name}_{seed}.json")
    out = json.load(open(name)) if os.path.exists(name) else {
        "model": model, "dataset": dataset, "input": input_name, "seed": seed, "n_train": int(len(tr)),
        "tests": {t: int(len(i)) for t, (i, _) in D["tests"].items()}, "sweep": {}}
    for n in budgets(cfg, dataset, D):
        if str(n) in out["sweep"]:
            continue
        m = M.fit(cfg, dataset, n, seed, build(P[tr], c0[tr], c1[tr], input_name, n), y[tr], D["nc"])
        r = {t: scores(y[idx], M.predict(m, build(*V[t], input_name, n))) for t, (idx, _) in D["tests"].items() if len(idx)}
        out["sweep"][str(n)] = r
        json.dump(out, open(name, "w"))
        print(f"[{model} {dataset} {input_name} {seed}] n={n} " + " ".join(f"{t}={v['macro_f1']:.4f}" for t, v in r.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    print(f"[{model} {dataset} {input_name} {seed}] done ({time.time() - t0:.0f}s)")
