"""Robustness numbers for the optimizer x ICL study, from the runs found in icl/results/data/runs.

    python3 icl/results/robust/robust.py        (from the repository root)

Writes robust.json and robust.md beside this file; prints the runs used per setting.  Every difference is taken seed
by seed over the seeds both settings have; n is whatever has finished (a run counts when it has a probe at its last
step).  ICL = mean accuracy (%) over the 14 never-seen pools (why.unseen) at the last checkpoint, unless a section says
otherwise.  Definitions of pools, rarest/commonest fifth and validation top-1 are why.py's.
"""
import glob
import itertools
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.getcwd())
from icl import why  # noqa: E402
from icl.seeds import interval, prediction  # noqa: E402

D = "icl/results/data/runs"
OUT = "icl/results/robust"
STEMS = {"A": "nodog-adamw-lr1e-3", "M": "nodog-muon-lr3e-3", "S": "nodog-sgd-lr1",
         "An": "nodog-adamw-lr1e-3-nopl", "Mn": "nodog-muon-lr3e-3-nopl",
         "AM": "nodog-adamw-lr1e-3-plmuon", "MA": "nodog-muon-lr1e-3-pladamw", "MM1": "nodog-muon-lr1e-3"}
RULES = ("plain", "white")
LAYERS = ["embed"] + [f"block{i}" for i in range(1, 13)] + ["gap", "pre_act", "pre_logits"]
BLOCKS = LAYERS[:13]
O365 = ["control"]                                  # the held-out Objects365 pool; the others are ImageNet-derived
U = why.UNSEEN_SET


# ---------------------------------------------------------------- loading
def load_all():
    G = {k: {} for k in STEMS}
    for p in sorted(glob.glob(os.path.join(D, "nodog-*"))):
        name = os.path.basename(p)
        if name.startswith("check-"):
            continue
        m = re.match(r"^(.*)-s(\d+)$", name)
        if not m:
            continue
        for k, stem in STEMS.items():
            if m.group(1) == stem:
                r = why.load(p)
                if why.probed(r):
                    lp = os.path.join(p, "linear.json")
                    r["linear"] = json.load(open(lp)) if os.path.exists(lp) else None
                    G[k][int(m.group(2))] = r
    return G


# ---------------------------------------------------------------- measures: fn(run) -> float | None
def safe(fn):
    def g(r):
        try:
            v = fn(r)
        except (KeyError, IndexError, TypeError):
            return None
        return None if v is None else float(v)
    return g


def ck_of(run, step):
    if step is None:
        return why.last(run)
    return next((c for c in run["probe"]["checkpoints"] if c["step"] == step), None)


def P(rule, layer, pools=None, step=None):
    """Probe accuracy (%), k = 5, mean over the unseen pools (or the given pools)."""
    def f(r):
        c = ck_of(r, step)
        k = str(r["probe"]["shots"][-1])
        ps = pools if pools is not None else why.unseen(r)
        rows = [c["fewshot"][rule][layer][p][k]["acc"] for p in ps]
        return 100 * float(np.mean(rows))
    return safe(f)


def F(layer, k, pools=None):
    """fewshot.json (plain rule) accuracy (%) at k shots at the last checkpoint, mean over the unseen pools."""
    def f(r):
        c = r["fewshot"]["checkpoints"][-1]
        if c["step"] != r["total_steps"]:
            return None
        ps = pools if pools is not None else why.unseen(r)
        return 100 * float(np.mean([c["results"][layer][p][str(k)]["acc"] for p in ps]))
    return safe(f)


def VAL(r):
    return 100 * why.last(r)["val"]["top1"]


def FIFTH(i):
    return lambda r: 100 * why.by_frequency(r, 5)[i]["top1"]


VAL, RARE, COMMON = safe(VAL), safe(FIFTH(0)), safe(FIFTH(4))


def L(tag, layer, k, pools=None):
    """linear.json: tag 'plain', 'white' or 'linear:<lam>'."""
    def f(r):
        res = r["linear"]["results"][layer]
        ps = pools if pools is not None else why.unseen(r)
        vals = []
        for p in ps:
            e = res.get(p, {}).get(str(k))
            if e is None:
                continue
            vals.append(e["linear"][tag[7:]]["acc"] if tag.startswith("linear:") else e[tag]["acc"])
        return 100 * float(np.mean(vals)) if vals else None
    return safe(f)


def sub(f, g):
    def h(r):
        a, b = f(r), g(r)
        return None if a is None or b is None else a - b
    return h


# ---------------------------------------------------------------- statistics
def fl(v):
    return None if v is None else float(v)


def within(S, fn):
    per = {s: fn(r) for s, r in sorted(S.items())}
    per = {s: v for s, v in per.items() if v is not None}
    v = list(per.values())
    return {"mean": fl(np.mean(v)) if v else None, "hw": fl(interval(v)) if len(v) > 1 else None, "n": len(v),
            "per_seed": per}


def perm_p(x, y):
    pool = list(x) + list(y)
    n, N = len(x), len(x) + len(y)
    obs = abs(np.mean(y) - np.mean(x))
    tot = hit = 0
    for idx in itertools.combinations(range(N), n):
        s = set(idx)
        a = [pool[i] for i in idx]
        b = [pool[i] for i in range(N) if i not in s]
        tot += 1
        hit += abs(np.mean(b) - np.mean(a)) >= obs - 1e-12
    return hit / tot, 2 / tot


def diff(X, Y, fx, fy=None, W=False):
    """Y - X seed by seed over the seeds both have: fy(Y[s]) - fx(X[s])."""
    fy = fy or fx
    xs, ys = {}, {}
    for s in sorted(set(X) & set(Y)):
        a, b = fx(X[s]), fy(Y[s])
        if a is not None and b is not None:
            xs[s], ys[s] = a, b
    d = {s: ys[s] - xs[s] for s in xs}
    v = list(d.values())
    n = len(v)
    out = {"mean": fl(np.mean(v)) if n else None, "hw": fl(interval(v)) if n > 1 else None, "n": n,
           "per_seed": d, "x": xs, "y": ys}
    if n:
        out["n_same_sign"] = int(sum(np.sign(t) == np.sign(np.mean(v)) for t in v))
    if W:
        out["welch_hw"] = out["perm_p"] = out["perm_p_min"] = None
        if n > 1:
            vx, vy = np.var(list(xs.values()), ddof=1), np.var(list(ys.values()), ddof=1)
            se = math.sqrt((vx + vy) / n)
            if se > 0:
                df = (vx / n + vy / n) ** 2 / ((vx / n) ** 2 / (n - 1) + (vy / n) ** 2 / (n - 1))
                out["welch_hw"] = float(stats.t.ppf(0.975, df) * se)
            out["perm_p"], out["perm_p_min"] = perm_p(list(xs.values()), list(ys.values()))
    return out


def from_dicts(d):
    """A stat from {seed: value} (already differences)."""
    v = list(d.values())
    n = len(v)
    out = {"mean": fl(np.mean(v)) if n else None, "hw": fl(interval(v)) if n > 1 else None, "n": n, "per_seed": d}
    if n:
        out["n_same_sign"] = int(sum(np.sign(t) == np.sign(np.mean(v)) for t in v))
    return out


def lead_change(with_layer, without):
    """One comparison minus another, seed by seed over the seeds both have (both are results of diff())."""
    return from_dicts({s: with_layer["per_seed"][s] - without["per_seed"][s]
                       for s in with_layer["per_seed"] if s in without["per_seed"]})


# ---------------------------------------------------------------- markdown
MD = []


def f2(v):
    return "-" if v is None else f"{v:.2f}"


def table(title, caption, rows):
    """rows: [(label, stat)]"""
    MD.append(f"\n**{title}** {caption}\n")
    W = any("welch_hw" in s for _, s in rows if s)
    MD.append("| item | mean | 95% half-width | n | same sign | per seed |" + (" Welch half-width | perm. p (min p) |" if W else ""))
    MD.append("|---|---|---|---|---|---|" + ("---|---|" if W else ""))
    for lab, s in rows:
        if s is None:
            continue
        ps = ", ".join(f2(v) for v in s["per_seed"].values())
        line = f"| {lab} | {f2(s['mean'])} | {f2(s['hw'])} | {s['n']} | {s.get('n_same_sign', '-')}/{s['n']} | {ps} |"
        if W:
            if "welch_hw" in s:
                p, pm = s["perm_p"], s["perm_p_min"]
                line += f" {f2(s['welch_hw'])} | {'-' if p is None else f'{p:.3f}'} ({'-' if pm is None else f'{pm:.3f}'}) |"
            else:
                line += " - | - |"
        MD.append(line)


# ---------------------------------------------------------------- main
def main():
    G = load_all()
    A, M, S, An, Mn = (G[k] for k in ("A", "M", "S", "An", "Mn"))
    res = {"runs": {k: sorted(v) for k, v in G.items()}, "n": {k: len(v) for k, v in G.items()}}
    MD.append("# Robustness numbers\n")
    MD.append("Runs used (seeds with a probe at the last step):\n")
    for k, stem in STEMS.items():
        MD.append(f"- {k} `{stem}`: seeds {sorted(G[k])} (n = {len(G[k])})")
    MD.append("\nICL = mean 5-shot accuracy (%) over the 14 never-seen pools at the last checkpoint. Differences are "
              "taken seed by seed over the seeds both settings have. Half-width = 95 % interval (Student t, n-1 df). "
              "Welch = unpaired interval; perm. p = exact two-sided permutation p over all splits of the pooled "
              "per-seed values, with the smallest p n allows (2 / C(2n, n)) in brackets. Objects365 held-out pool = "
              "`control`; the other 13 pools are ImageNet-derived.")
    pools_all = why.unseen(next(iter(A.values())))
    res["pools"] = pools_all
    pools_in = [p for p in pools_all if p not in O365]

    def step(rule, pools=None):
        return sub(P(rule, "pre_logits", pools), P(rule, "gap", pools))

    def stepF(k):
        return sub(F("pre_logits", k), F("gap", k))

    comps = (("M-A", M), ("S-A", S))

    # ---------------- section 1
    s1 = res["s1"] = {}
    # a
    MD.append("\n## 1. Where the difference is made")
    a = s1["a"] = {}
    rows = []
    for cn, Y in comps:
        for layer in ("pre_logits", "gap"):
            for rule in RULES:
                st = a[f"{cn} {layer} {rule} probe k5"] = diff(A, Y, P(rule, layer), W=True)
                rows.append((f"{cn} {layer} {rule} (k=5)", st))
            for k in (1, 5, 10):
                st = a[f"{cn} {layer} plain fewshot k{k}"] = diff(A, Y, F(layer, k), W=True)
                rows.append((f"{cn} {layer} plain fewshot k={k}", st))
    table("1a", "Difference in ICL at the layer.", rows)
    # b
    b = s1["b"] = {"step": {}, "diff": {}}
    rows = []
    stepfns = {**{f"{rule} probe k5": step(rule) for rule in RULES}, **{f"plain fewshot k{k}": stepF(k) for k in (1, 5, 10)}}
    for sn, SG in (("A", A), ("M", M), ("S", S)):
        for tn, fn in stepfns.items():
            st = b["step"][f"{sn} {tn}"] = within(SG, fn)
            rows.append((f"{sn} step {tn}", st))
    table("1b-i", "Step across the layer: accuracy at `pre_logits` minus at `gap`, within each run.", rows)
    rows = []
    for cn, Y in comps:
        for tn, fn in stepfns.items():
            st = b["diff"][f"{cn} {tn}"] = diff(A, Y, fn, W=True)
            rows.append((f"{cn} step {tn}", st))
    table("1b-ii", "Difference of the step between settings.", rows)
    # c
    c = s1["c"] = {}
    for cn, Y in comps:
        for rule in RULES:
            ov = b["diff"][f"{cn} {rule} probe k5"]["mean"]
            per = {p: diff(A, Y, step(rule, [p])) for p in pools_all}
            same = sum(1 for s in per.values() if s["mean"] is not None and np.sign(s["mean"]) == np.sign(ov))
            clear = sum(1 for s in per.values() if s["hw"] is not None and abs(s["mean"]) > s["hw"])
            clear_same = sum(1 for s in per.values() if s["hw"] is not None and abs(s["mean"]) > s["hw"]
                             and np.sign(s["mean"]) == np.sign(ov))
            fam = {"objects365": diff(A, Y, step(rule, O365)), "imagenet_derived": diff(A, Y, step(rule, pools_in))}
            c[f"{cn} {rule}"] = {"overall_mean": ov, "pools": per, "n_pools": len(per), "n_same_sign_as_overall": same,
                                 "n_clear": clear, "n_clear_same_sign": clear_same, "families": fam}
            rows = [(p, per[p]) for p in pools_all] + [("Objects365 held-out (control)", fam["objects365"]),
                                                       (f"ImageNet-derived ({len(pools_in)} pools)", fam["imagenet_derived"])]
            table(f"1c {cn} {rule}", f"Difference of the step by pool (k=5). Pools with the overall sign: {same}/{len(per)}; "
                  f"clear (|mean| > half-width): {clear}/{len(per)}, of them with the overall sign: {clear_same}.", rows)
    # d
    d = s1["d"] = {}
    for cn, Y in comps:
        for rule in RULES:
            per = {ly: diff(A, Y, P(rule, ly)) for ly in LAYERS}
            valid = {ly: per[ly] for ly in BLOCKS if per[ly]["mean"] is not None}
            big = max(valid, key=lambda ly: abs(valid[ly]["mean"])) if valid else None
            d[f"{cn} {rule}"] = {"layers": per, "largest_abs_embed_to_block12": big,
                                 "largest": None if big is None else {"layer": big, **{k: v for k, v in per[big].items() if k in ("mean", "hw", "n")}}}
            table(f"1d {cn} {rule}", "Difference in ICL after each layer (k=5). Largest |difference| among embed..block12: "
                  + (f"{big} ({f2(per[big]['mean'])} ± {f2(per[big]['hw'])})" if big else "-") + ".",
                  [(ly, per[ly]) for ly in LAYERS])
    # e
    e = s1["e"] = {}
    for cn, Y in comps:
        common = sorted(set(A) & set(Y))
        if not common:
            continue
        sets = [set(c_["step"] for c_ in (A if False else G_)[s]["probe"]["checkpoints"]) for G_ in (A, Y) for s in common]
        steps = sorted(set.intersection(*sets))
        tot = A[common[0]]["total_steps"]
        for rule in RULES:
            per = {}
            for stp in steps:
                per[str(stp)] = {"pct": 100 * stp / tot, **diff(A, Y, sub(P(rule, "pre_logits", None, stp), P(rule, "gap", None, stp)))}
            e[f"{cn} {rule}"] = per
            table(f"1e {cn} {rule}", "Difference of the step (k=5) at every checkpoint the compared runs share (item = % of training).",
                  [(f"{v['pct']:.2f}%", v) for v in per.values()])
    # f
    f = s1["f"] = {}
    f["Mn-An gap"] = {rule: diff(An, Mn, P(rule, "gap"), W=True) for rule in RULES}
    f["M-A pre_logits"] = {rule: a[f"M-A pre_logits {rule} probe k5"] for rule in RULES}
    f["M-A gap"] = {rule: a[f"M-A gap {rule} probe k5"] for rule in RULES}
    # Muon's lead at the last hidden layer of the standard model minus its lead in the model without the layer
    f["lead change"] = {rule: lead_change(f["M-A pre_logits"][rule], f["Mn-An gap"][rule]) for rule in RULES}
    rows = [(f"Mn-An gap {r_}", f["Mn-An gap"][r_]) for r_ in RULES] + \
           [(f"M-A pre_logits {r_}", f["M-A pre_logits"][r_]) for r_ in RULES] + [(f"M-A gap {r_}", f["M-A gap"][r_]) for r_ in RULES]
    table("1f-i", "Without the layer: Mn - An at `gap`, beside M - A.", rows)
    f["removal"] = {}
    rows = []
    for nm, base, other in (("An-A", A, An), ("Mn-M", M, Mn)):
        for layer in ("pre_logits", "gap"):
            for rule in RULES:
                st = f["removal"][f"{nm} vs standard {layer} {rule}"] = diff(base, other, P(rule, layer), P(rule, "gap"), W=True)
                rows.append((f"{nm}: no-layer gap minus standard {layer} {rule}", st))
    table("1f-ii", "What removing the layer gives each optimizer: no-layer `gap` minus the standard model at the named layer (k=5).", rows)
    # g
    lin = [r for G_ in (A, M, S, An, Mn) for r in G_.values() if r["linear"]]
    if lin:
        shots = sorted({int(k) for r in lin for k in r["linear"]["shots"]})
        lamkeys = []
        for r in lin:
            for ly in r["linear"]["results"].values():
                for po in ly.values():
                    for ke in po.values():
                        for lk in ke.get("linear", {}):
                            if lk not in lamkeys:
                                lamkeys.append(lk)
        lamkeys.sort(key=float)
        tests = [(f"{t} k{k}", t, k) for t in RULES for k in [25] if k in shots] + \
                [(f"linear lam={lk} k{k}", f"linear:{lk}", k) for lk in lamkeys for k in shots]
        g = s1["g"] = {}
        rowsa, rowsb, rowsf = [], [], []
        for nm, tag, k in tests:
            ent = g[nm] = {"a": {}, "b": {}, "f": {}}
            for cn, Y in comps:
                for layer in ("pre_logits", "gap"):
                    ent["a"][f"{cn} {layer}"] = diff(A, Y, L(tag, layer, k), W=True)
                    rowsa.append((f"{nm} {cn} {layer}", ent["a"][f"{cn} {layer}"]))
            stp = sub(L(tag, "pre_logits", k), L(tag, "gap", k))
            for sn, SG in (("A", A), ("M", M), ("S", S)):
                ent["b"][f"{sn} step"] = within(SG, stp)
                rowsb.append((f"{nm} {sn} step", ent["b"][f"{sn} step"]))
            for cn, Y in comps:
                ent["b"][f"{cn} step diff"] = diff(A, Y, stp, W=True)
                rowsb.append((f"{nm} {cn} step diff", ent["b"][f"{cn} step diff"]))
            ent["f"]["Mn-An gap"] = diff(An, Mn, L(tag, "gap", k), W=True)
            rowsf.append((f"{nm} Mn-An gap", ent["f"]["Mn-An gap"]))
            for bn, base, other in (("An-A", A, An), ("Mn-M", M, Mn)):
                for layer in ("pre_logits", "gap"):
                    ent["f"][f"{bn} vs {layer}"] = diff(base, other, L(tag, layer, k), L(tag, "gap", k), W=True)
                    rowsf.append((f"{nm} {bn}: no-layer gap minus standard {layer}", ent["f"][f"{bn} vs {layer}"]))
        table("1g-a", "linear.json (last checkpoint, accuracy in %): difference at the layer.", rowsa)
        table("1g-b", "linear.json: step across the layer, and the difference of steps.", rowsb)
        table("1g-f", "linear.json: without the layer.", rowsf)

    # ---------------- section 2
    MD.append("\n## 2. The first 2 % of training")
    s2 = res["s2"] = {}
    for sn, SG in (("A", A), ("M", M), ("S", S)):
        o = s2[sn] = {"seeds": {}}
        for sd, r in sorted(SG.items()):
            cks = [c_ for c_ in r["probe"]["checkpoints"] if c_["step"] <= 0.02 * r["total_steps"]]
            rowsck = []
            for c_ in cks:
                st = c_["step"]
                rowsck.append({"step": st, "pct": 100 * st / r["total_steps"],
                               "icl_pre_logits_plain": P("plain", "pre_logits", None, st)(r),
                               "icl_pre_logits_white": P("white", "pre_logits", None, st)(r),
                               "tanh_over90_pct": safe(lambda r_, c_=c_: 100 * c_["tanh"][U]["over90"])(r),
                               "tanh_over99_pct": safe(lambda r_, c_=c_: 100 * c_["tanh"][U]["over99"])(r),
                               "pre_logits_stable_rank": safe(lambda r_, c_=c_: c_["weights"]["pre_logits.weight"]["srank"])(r)})
            sr = [(c_["step"], c_["weights"]["pre_logits.weight"]["srank"]) for c_ in r["probe"]["checkpoints"]
                  if "pre_logits.weight" in c_["weights"]]
            low = min(sr, key=lambda t: t[1])
            ent = {"checkpoints": rowsck, "stable_rank_lowest": low[1], "stable_rank_lowest_step": low[0],
                   "stable_rank_end": sr[-1][1], "stable_rank_start": sr[0][1]}
            for rule in RULES:
                v = [x[f"icl_pre_logits_{rule}"] for x in rowsck]
                base = v[0]
                lo = min(range(1, len(v)), key=lambda i: v[i]) if len(v) > 1 else None
                ent[f"icl_{rule}"] = {"step0": base, "lowest_after": None if lo is None else v[lo],
                                      "lowest_step": None if lo is None else rowsck[lo]["step"],
                                      "drop_vs_step0": None if lo is None else v[lo] - base,
                                      "falls_below": bool(lo is not None and v[lo] < base)}
            o["seeds"][str(sd)] = ent
        for rule in RULES:
            ds = [e_[f"icl_{rule}"] for e_ in o["seeds"].values()]
            o[f"{rule}_n_seeds_below_step0"] = int(sum(x["falls_below"] for x in ds))
            o[f"{rule}_n"] = len(ds)
            dv = [x["drop_vs_step0"] for x in ds if x["drop_vs_step0"] is not None]
            o[f"{rule}_mean_drop"] = fl(np.mean(dv)) if dv else None
        MD.append(f"\n**2 {sn}** (checkpoints up to 2 % of the steps; saturated = share of `pre_logits` tanh units with |value| > 0.9 on the never-seen ImageNet images, `over90`; stable rank of the `pre_logits` matrix)\n")
        MD.append("| seed | step | % | ICL plain | ICL white | saturated % | stable rank |")
        MD.append("|---|---|---|---|---|---|---|")
        for sd, e_ in o["seeds"].items():
            for x in e_["checkpoints"]:
                MD.append(f"| {sd} | {x['step']} | {x['pct']:.2f} | {f2(x['icl_pre_logits_plain'])} | {f2(x['icl_pre_logits_white'])} | {f2(x['tanh_over90_pct'])} | {f2(x['pre_logits_stable_rank'])} |")
        MD.append("\n| seed | plain: step 0, lowest after, drop, below? | white: step 0, lowest after, drop, below? | stable rank: lowest (step), end |")
        MD.append("|---|---|---|---|")
        for sd, e_ in o["seeds"].items():
            t = lambda q: f"{f2(q['step0'])}, {f2(q['lowest_after'])}, {f2(q['drop_vs_step0'])}, {'yes' if q['falls_below'] else 'no'}"
            MD.append(f"| {sd} | {t(e_['icl_plain'])} | {t(e_['icl_white'])} | {f2(e_['stable_rank_lowest'])} ({e_['stable_rank_lowest_step']}), {f2(e_['stable_rank_end'])} |")
        MD.append(f"\n{sn}: seeds below step 0 within 2 %: plain {o['plain_n_seeds_below_step0']}/{o['plain_n']} (mean drop {f2(o['plain_mean_drop'])}), "
                  f"white {o['white_n_seeds_below_step0']}/{o['white_n']} (mean drop {f2(o['white_mean_drop'])}).")

    # ---------------- section 3
    MD.append("\n## 3. The 2x2 at one learning rate")
    s3 = res["s3"] = {}
    cells = {"A": A, "AM": G["AM"], "MA": G["MA"], "MM1": G["MM1"]}
    common = sorted(set.intersection(*(set(v) for v in cells.values())))
    s3["seeds"] = common
    s3["n"] = len(common)
    MD.append(f"\nCells (blocks' optimizer first, matrix's second): A = AdamW/AdamW, AM = AdamW blocks/Muon matrix, MA = Muon blocks/AdamW matrix, MM1 = Muon/Muon at 1e-3. Seeds all four have: {common} (n = {len(common)}). Matrix effect = (AM-A, MM1-MA) and their average; blocks effect = (MA-A, MM1-AM) and their average; interaction = (MM1-MA) - (AM-A).")
    meas = {"val": VAL, "rarest fifth": RARE, "ICL pre_logits plain": P("plain", "pre_logits"), "ICL pre_logits white": P("white", "pre_logits"),
            "ICL gap plain": P("plain", "gap"), "ICL gap white": P("white", "gap")}
    for mn, fn in meas.items():
        v = {c_: {s: fn(cells[c_][s]) for s in common} for c_ in cells}
        ok = [s for s in common if all(v[c_][s] is not None for c_ in cells)]
        o = {"cell_means": {c_: (fl(np.mean([v[c_][s] for s in ok])) if ok else None) for c_ in cells},
             "cell_values": {c_: {str(s): v[c_][s] for s in ok} for c_ in cells}, "n": len(ok)}
        def eff(f_):
            return from_dicts({s: f_(lambda c_: v[c_][s]) for s in ok})
        o["matrix_under_adamw_blocks (AM-A)"] = eff(lambda x: x("AM") - x("A"))
        o["matrix_under_muon_blocks (MM1-MA)"] = eff(lambda x: x("MM1") - x("MA"))
        o["matrix_effect_average"] = eff(lambda x: ((x("AM") - x("A")) + (x("MM1") - x("MA"))) / 2)
        o["blocks_under_adamw_matrix (MA-A)"] = eff(lambda x: x("MA") - x("A"))
        o["blocks_under_muon_matrix (MM1-AM)"] = eff(lambda x: x("MM1") - x("AM"))
        o["blocks_effect_average"] = eff(lambda x: ((x("MA") - x("A")) + (x("MM1") - x("AM"))) / 2)
        o["interaction"] = eff(lambda x: (x("MM1") - x("MA")) - (x("AM") - x("A")))
        # is the matrix's effect larger than the blocks'?  (their difference, seed by seed)
        o["matrix_minus_blocks"] = eff(lambda x: ((x("AM") - x("A")) + (x("MM1") - x("MA"))) / 2
                                       - ((x("MA") - x("A")) + (x("MM1") - x("AM"))) / 2)
        s3[mn] = o
        MD.append(f"\n**3 {mn}** cell means (n = {o['n']}): " + ", ".join(f"{c_} {f2(m_)}" for c_, m_ in o["cell_means"].items()) + "\n")
        table(f"3 {mn}", "effects.", [(k, o[k]) for k in o if isinstance(o[k], dict) and "per_seed" in o[k]])

    # is the gain divided differently between the two parts on unseen classes than on trained ones?  (the matrix's
    # effect minus the blocks', on ICL at pre_logits, minus the same on validation; seed by seed)
    s3["split"] = {rule: lead_change(s3[f"ICL pre_logits {rule}"]["matrix_minus_blocks"], s3["val"]["matrix_minus_blocks"])
                   for rule in RULES}
    # what crossing the layer costs in each cell (pre_logits minus gap, within each run)
    s3["step"] = {rule: {c_: from_dicts({s_: s3[f"ICL pre_logits {rule}"]["cell_values"][c_][s_] - s3[f"ICL gap {rule}"]["cell_values"][c_][s_]
                                         for s_ in s3[f"ICL pre_logits {rule}"]["cell_values"][c_]}) for c_ in cells}
                  for rule in RULES}
    table("3 contrasts", "matrix's effect minus blocks' effect per measure; `split` = that contrast on ICL minus on validation; step = pre_logits minus gap per cell.",
          [(f"{mn} matrix minus blocks", s3[mn]["matrix_minus_blocks"]) for mn in meas]
          + [(f"split {rule}", s3["split"][rule]) for rule in RULES]
          + [(f"step {rule} {c_}", s3["step"][rule][c_]) for rule in RULES for c_ in cells])

    # ---------------- section 6: the seeds added on 2026-11-22 against the first three of their setting
    MD.append("\n## 6. New seeds against the first three")
    s6 = res["s6"] = {"rows": {}}
    m6 = {"validation top-1": VAL, "rarest fifth": RARE, "ICL pre_logits plain": P("plain", "pre_logits"),
          "ICL pre_logits white": P("white", "pre_logits"), "ICL gap plain": P("plain", "gap"), "ICL gap white": P("white", "gap")}
    inside = total = 0
    for sn, Sx in (("A", A), ("M", M), ("S", S), ("An", An), ("Mn", Mn)):
        for mn, fn in m6.items():
            if sn in ("An", "Mn") and "pre_logits" in mn:
                continue
            old = [fn(Sx[k]) for k in sorted(Sx) if k <= 2]
            new = {str(k): fn(Sx[k]) for k in sorted(Sx) if k >= 3}
            if len(old) != 3 or any(v is None for v in old) or not new:
                continue
            mean, hw = fl(np.mean(old)), fl(prediction(old))
            ok = {k: bool(abs(v - mean) <= hw) for k, v in new.items()}
            s6["rows"][f"{sn} {mn}"] = {"old_mean": mean, "prediction_hw": hw, "old": old, "new": new, "inside": ok}
            inside += sum(ok.values()); total += len(ok)
    s6["inside"], s6["total"] = inside, total
    MD.append(f"\nOf {total} values of the new seeds (five settings, up to six measures), {inside} lie inside the range that the first three seeds of the setting predict for one more seed (mean ± t·sd·√(1+1/3)).\n")
    MD.append("| setting and measure | first three: mean | ± prediction | new seeds | inside |\n|---|---|---|---|---|")
    for k_, r_ in s6["rows"].items():
        MD.append(f"| {k_} | {f2(r_['old_mean'])} | {f2(r_['prediction_hw'])} | " + ", ".join(f2(v) for v in r_["new"].values()) + " | " + ", ".join("yes" if v else "NO" for v in r_["inside"].values()) + " |")

    # ---------------- section 4
    MD.append("\n## 4. Rare classes and the layer")
    s4 = res["s4"] = {}
    for mn, fn in (("rarest fifth", RARE), ("validation top-1", VAL), ("commonest fifth", COMMON)):
        s4[mn] = {"M-A": diff(A, M, fn), "Mn-An": diff(An, Mn, fn, W=True), "An-A": diff(A, An, fn, W=True), "Mn-M": diff(M, Mn, fn, W=True)}
        # how much of Muon's lead goes with the layer: the lead with it minus the lead without it, seed by seed
        s4[mn]["lead change"] = lead_change(s4[mn]["M-A"], s4[mn]["Mn-An"])
        table(f"4 {mn}", "validation accuracy (%).", list(s4[mn].items()))

    # ---------------- section 5
    MD.append("\n## 5. SGD")
    s5 = res["s5"] = {}
    for mn, fn in (("validation top-1", VAL), ("rarest fifth", RARE), ("ICL pre_logits plain", P("plain", "pre_logits")),
                   ("ICL pre_logits white", P("white", "pre_logits")), ("ICL gap plain", P("plain", "gap")),
                   ("ICL gap white", P("white", "gap"))):
        s5[mn] = diff(A, S, fn, W=True)
    table("5", "S - A.", list(s5.items()))

    with open(os.path.join(OUT, "robust.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    with open(os.path.join(OUT, "robust.md"), "w") as fh:
        fh.write("\n".join(MD) + "\n")
    for k, stem in STEMS.items():
        print(f"{k:4s} {stem}: seeds {sorted(G[k])}")
    print(os.path.join(OUT, "robust.json"), os.path.join(OUT, "robust.md"))


if __name__ == "__main__":
    main()
