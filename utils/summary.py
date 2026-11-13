import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from utils.config import strs, ints, path


def collect(cfg, model):
    rows = defaultdict(list)
    for f in glob.glob(os.path.join(path(cfg, "outputs_dir"), "results", model, "*", "*.json")):
        d = json.load(open(f))
        for n, r in d["sweep"].items():
            for cond, v in r.items():
                if isinstance(v, dict):
                    rows[(d["dataset"], cond, d["input"], int(n))].append(v["macro_f1"])
    return rows


def write_table(cfg, model, rows, ns):
    out = os.path.join(path(cfg, "outputs_dir"), "summary", f"table_{model}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "condition", "input", "n", "macro_f1_mean", "macro_f1_std", "runs"])
        for (ds, cond, inp, n), xs in sorted(rows.items()):
            if n in ns:
                w.writerow([ds, cond, inp, n, round(statistics.mean(xs), 4), round(statistics.pstdev(xs), 4) if len(xs) > 1 else 0.0, len(xs)])
    return out


def write_curves(cfg, model, rows):
    inputs_ = strs(cfg, "experiment", "inputs")
    keys = sorted({(ds, cond) for ds, cond, _, _ in rows})
    outs = []
    for ds, cond in keys:
        ns = sorted({n for d, c, _, n in rows if d == ds and c == cond})
        out = os.path.join(path(cfg, "outputs_dir"), "summary", f"curves_{model}_{ds}_{cond}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n"] + [f"{i}_{s}" for i in inputs_ for s in ("mean", "std")])
            for n in ns:
                row = [n]
                for i in inputs_:
                    xs = rows.get((ds, cond, i, n), [])
                    row += [round(statistics.mean(xs), 4) if xs else "", round(statistics.pstdev(xs), 4) if len(xs) > 1 else (0.0 if xs else "")]
                w.writerow(row)
        outs.append(out)
    return outs


def run(cfg, *models):
    os.makedirs(os.path.join(path(cfg, "outputs_dir"), "summary"), exist_ok=True)
    for model in models or ("lgbm", "cnn"):
        rows = collect(cfg, model)
        if rows:
            print(write_table(cfg, model, rows, {10, 30}))
            print(len(write_curves(cfg, model, rows)), "curve files")
