"""The numbers behind icl/results/page.html, read from icl/results/data (the probes and logs) and results/flops.

    python3 icl/results/page_data.py [OUT.json]      (from the repository root; fetch the data first: cluster/pull.sh
                                                      and run results/make3.sh for results/flops)

Writes the numbers as JSON when a path is given, and always rebuilds icl/results/page.html from page.src.html by
putting the numbers in place of the line `const DATA = null;`.

Nothing is computed here that why.py, seeds.py or flops.py do not define; this only gathers their numbers per
checkpoint, per block and per pool into one file for the page's figures.
"""
import glob
import re
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from icl import why  # noqa: E402

D = "icl/results/data/runs"
ATT, MLP = why.PARTS[0][1], why.PARTS[1][1]


def r1(v, n=1):
    """A number for the page.  The page rounds for display itself, so seven decimals are kept here: a value
    rounded twice (70.2489 -> 70.25 -> "70.3") can end one digit off."""
    return None if v is None else round(float(v), max(n, 7))


def label(r):
    return r["setting"] + (f", {r['epochs']:g} epochs" if r["epochs"] != 90 else "")


def series(by_seed, fn, n=1):
    got = why.along(by_seed, fn)
    if got is None:
        return None
    _, mean, lo, hi = got
    return {"m": [r1(v, n) for v in mean], "lo": [r1(v, n) for v in lo], "hi": [r1(v, n) for v in hi]}


def fs(rule, layer, group="unseen"):
    return lambda r, c: (lambda v: None if v is None else 100 * v)(why.acc(r, c, rule, layer, group))


def quint(i):
    return lambda r, c: (lambda v: None if v is None else 100 * v)(why.by_frequency(r, 5, c)[i]["top1"])


def block_stat(run, ck, fams, key, sub=None):
    return [why.in_block(ck, b, fams, key, sub) for b in why.block_numbers(run)]


def nice_variant(r):
    """A run's variant in the page's words (the trainer's own wording names layers and run folders)."""
    v = r["variant"]
    if v.startswith("from "):
        return "after " + ("Muon" if "-muon-" in v else "AdamW") + "'s first 10%"
    return {"no pre_logits": "no last hidden layer", "pre_logits by AdamW": "last matrix by AdamW",
            "on pre_logits only": "Muon on the last matrix only, AdamW on the blocks",
            "on pre_logits only at 0.003": "Muon on the last matrix only at 3e-3, AdamW on the blocks at 1e-3"}.get(v, v)


def padded(sv, n):
    """A series of a run that took over in mid-training, with nothing at the checkpoints it does not have."""
    return None if sv is None else {k: [None] * (n - len(v)) + v for k, v in sv.items()}


def mean_runs(rows):
    """Mean over runs of equally shaped nested lists; None where any run has None."""
    a = np.array(rows, dtype=float)
    m = a.mean(0)
    return np.where(np.isnan(m), None, np.round(m, 7)).tolist()


def slim(x, seeds=True):
    """robust.json for the page: a comparison (a dict with a mean and an n) in short keys, numbers rounded."""
    if isinstance(x, dict):
        if "mean" in x and "n" in x:
            e = {"m": r1(x["mean"], 7), "hw": r1(x.get("hw"), 7), "n": x["n"]}
            for a, b in (("n_same_sign", "same"), ("perm_p", "p"), ("perm_p_min", "pmin"), ("welch_hw", "w"),
                         ("pct", "pct"), ("layer", "layer")):
                if x.get(a) is not None:
                    e[b] = round(x[a], 4) if isinstance(x[a], float) else x[a]
            if seeds and isinstance(x.get("per_seed"), dict):
                e["s"] = {k: r1(v, 7) for k, v in x["per_seed"].items()}
            return e
        return {k: slim(v, seeds) for k, v in x.items() if k not in ("x", "y")}
    if isinstance(x, list):
        return [slim(v, seeds) for v in x]
    return round(x, 7) if isinstance(x, float) else x


def folded(page):
    """A caption with <!--more--> in it: what stands before stays under the chart, the rest folds away."""
    return re.sub(r'<p class="cap">((?:(?!</p>).)*?)\s*<!--more-->((?:(?!</p>).)*?)</p>',
                  r'<p class="cap">\1</p>\n<details class="more"><summary>More about this chart</summary><p class="cap">\2</p></details>',
                  page, flags=re.S)


def main(out):
    runs = [why.load(p) for p in sorted(glob.glob(os.path.join(D, "nodog-*")))]
    runs = [r for r in runs if why.probed(r)]
    groups = {}
    for r in runs:
        groups.setdefault(label(r), {})[tuple(r["seeds"])] = r
    A, M, S, M1 = (groups[k] for k in ("adamw lr 0.001", "muon lr 0.003", "sgd lr 1", "muon lr 0.001"))
    main4 = {"adamw": A, "muon3": M, "sgd": S, "muon1": M1}
    extra = {"mlp": groups["muon lr 0.001, on the MLP only"], "attn": groups["muon lr 0.001, on the attention only"],
             "pla": groups["muon lr 0.003, pre_logits by AdamW"], "m70": groups["muon lr 0.003, 70 epochs"]}
    # the runs of 2026-11-20 (one each, seed 0): the missing cell, the two that change optimizer at 10 % of training;
    # seed 0 of the two main settings, which those two go on from; and the settings finished on 2026-11-21: the model
    # without the last hidden layer (three seeds each), the mirror control (AdamW on the blocks, Muon on the last
    # hidden layer's matrix alone: three seeds at 1e-3, one run with that matrix at 3e-3)
    swap = {("m2a" if k.startswith("adamw") else "a2m"): g for k, g in groups.items() if ", from " in k}
    ANP, MNP = groups["adamw lr 0.001, no pre_logits"], groups["muon lr 0.003, no pre_logits"]
    PLA1, PLM = groups["muon lr 0.001, pre_logits by AdamW"], groups["muon lr 0.001, on pre_logits only"]
    PLM3 = groups["muon lr 0.001 / muon 0.003, on pre_logits only at 0.003"]
    new = {"pla1": PLA1, "anp": ANP, "mnp": MNP, "plm": PLM, "plm3": PLM3, **swap,
           "adamw0": {(0, 0): A[(0, 0)]}, "muon30": {(0, 0): M[(0, 0)]}}
    any_run = next(iter(A.values()))
    steps = [c["step"] for c in any_run["probe"]["checkpoints"]]
    data = {"pct": [round(100 * s / any_run["total_steps"], 2) for s in steps], "steps": steps}

    # along training
    U = why.UNSEEN_SET

    def dims(layer):
        """How many directions a layer's features fill on the unseen images (the entropy rank of probe.py)."""
        return lambda r, c: c["shape"].get(layer, {}).get(U, {}).get("erank")
    def has_pl(r):
        return "no pre_logits" not in r["variant"]
    data["along"] = {}
    for k, g in {**main4, **extra, **new}.items():
        data["along"][k] = {
            "val": series(g, lambda r, c: 100 * c["val"]["top1"]),
            "un_pl_w": series(g, fs("white", "pre_logits")), "un_gap_w": series(g, fs("white", "gap")),
            "un_pl_p": series(g, fs("plain", "pre_logits")), "un_gap_p": series(g, fs("plain", "gap")),
            "dogs_pl_p": series(g, fs("plain", "pre_logits", "dogs_5way")),
            "rare": series(g, quint(0)), "common": series(g, quint(4)),
            "sat": series(g, lambda r, c: 100 * c["tanh"][U]["over90"] if U in c["tanh"] and has_pl(r) else None),
            "plrank": series(g, lambda r, c: c["weights"].get("pre_logits.weight", {}).get("srank")),
            "dim_gap": series(g, dims("gap")), "dim_act": series(g, dims("pre_act")),
            "dim_pl": series(g, dims("pre_logits")),
        }
        data["along"][k] = {name: padded(sv, len(steps)) for name, sv in data["along"][k].items()}

    # the end of training, for the tables (the page rounds for display)
    def end_of(g, fn):
        got = why.along(g, fn)
        return None if got is None or got[1][-1] is None else round(float(got[1][-1]), 7)
    at_end = {"val": lambda r, c: 100 * c["val"]["top1"], "rare": quint(0), "common": quint(4),
              "dogs_pl_p": fs("plain", "pre_logits", "dogs_5way"), "un_gap_p": fs("plain", "gap"),
              "un_gap_w": fs("white", "gap"), "un_pl_p": fs("plain", "pre_logits"), "un_pl_w": fs("white", "pre_logits")}
    data["end"] = {k: {m: end_of(g, fn) for m, fn in at_end.items()} for k, g in {**main4, **extra, **new}.items()}

    # the end of training: every run's validation top-1, by optimizer and learning rate
    data["final"] = [{"opt": r["optimizer"], "lr": r["lr"], "seed": r["seeds"][0], "variant": nice_variant(r),
                      "epochs": r["epochs"], "val": r1(100 * why.last(r)["val"]["top1"], 2),
                      "un_pl_w": r1(fs("white", "pre_logits")(r, why.last(r)), 2),
                      "un_gap_w": r1(fs("white", "gap")(r, why.last(r)), 2)} for r in runs]

    # class-frequency fifths at the end
    q = why.by_frequency(any_run, 5)
    data["quint"] = {"range": [[x["fewest"], x["most"]] for x in q]}
    for k, g in main4.items():
        data["quint"][k] = [r1(np.mean([100 * why.by_frequency(r, 5)[i]["top1"] for r in g.values()])) for i in range(5)]
    data["quint_diff"] = {}
    for k in ("muon3", "sgd"):
        data["quint_diff"][k] = [[r1(x, 2) for x in why.compare(A, main4[k], lambda r, i=i: 100 * why.by_frequency(r, 5)[i]["top1"])]
                                 for i in range(5)]

    # the 14 unseen pools at the end, against AdamW
    pools = why.unseen(any_run)
    data["pools"] = {"names": pools, "ways": [any_run["probe"]["pools"][p]["n_way"] for p in pools]}
    for k in ("muon3", "sgd"):
        for rule in ("plain", "white"):
            for layer in ("gap", "pre_logits"):
                rows = [why.compare(A, main4[k], lambda r, p=p: fs(rule, layer, p)(r, why.last(r))) for p in pools]
                data["pools"][f"{k}_{rule}_{layer}"] = [[r1(x, 2) for x in row] for row in rows]
    for rule in ("plain", "white"):
        for layer in ("gap", "pre_logits"):
            data["pools"][f"adamw_{rule}_{layer}"] = [r1(np.mean([fs(rule, layer, p)(r, why.last(r)) for r in A.values()]))
                                                      for p in pools]

    # the model without the last hidden layer, test by test: Muon minus AdamW, seed by seed over three seeds
    # ([difference, half-width of its 95 % interval])
    def versus(base, other, fn, fn_base=None):
        """other minus base at the end of training as [difference, half-width]: seed by seed where both have several
        seeds, else the one run against the scatter of the base's seeds.  fn_base reads the base at another place."""
        if fn_base is None:
            got = why.compare(base, other, lambda r: fn(r, why.last(r)))
        else:
            d = [fn(other[s], why.last(other[s])) - fn_base(base[s], why.last(base[s])) for s in sorted(base) if s in other]
            got = (float(np.mean(d)), why.interval(d))
        return [r1(got[0], 2), r1(got[1], 2)]
    data["nopl"] = {rule: [versus(ANP, MNP, fs(rule, "gap", p)) for p in pools] for rule in ("plain", "white")}

    # exact differences for the page's notes (the series above are rounded to one decimal).  lead: Muon 3e-3 minus
    # AdamW 1e-3 at the end, [with the last hidden layer (means of three runs), without it (one run each)].
    # steps: the few-shot accuracy at the last hidden layer of the settings that differ by one thing each.
    def at_end(g, fn):
        return float(np.mean([fn(r, why.last(r)) for r in g.values()]))
    measures = {"val": lambda r, c: 100 * c["val"]["top1"], "rare": quint(0), "common": quint(4),
                **{f"un_{layer[:3] if layer == 'gap' else 'pl'}_{rule[0]}": fs(rule, layer)
                   for rule in ("plain", "white") for layer in ("gap", "pre_logits")}}
    data["lead"] = {m: {"w": versus(A, M, fn), "o": versus(ANP, MNP, fn)} for m, fn in measures.items()}
    # the scoreboard under the answer: every seed of AdamW 1e-3 and Muon 3e-3 at the end of training, in the normal
    # model and in the one built without the last hidden layer (np_*), with Muon minus AdamW seed by seed
    def seeds_of(g, fn):
        return [round(float(fn(r, why.last(r))), 2) for _, r in sorted(g.items())]
    def pair(a, m, fn):
        d, hw = versus(a, m, fn)
        return {"a": seeds_of(a, fn), "m": seeds_of(m, fn), "d": d, "hw": hw}
    data["score"] = {m: pair(A, M, fn) for m, fn in measures.items()}
    data["score"].update({"np_val": pair(ANP, MNP, measures["val"]), "np_rare": pair(ANP, MNP, measures["rare"]),
                          "np_w": pair(ANP, MNP, fs("white", "gap")), "np_p": pair(ANP, MNP, fs("plain", "gap"))})
    # SGD 1.0 beside them in the normal model (no SGD run was made without the layer): its seeds, and SGD minus
    # AdamW seed by seed
    # (three decimals, so that the page's one-decimal figures are rounded once, not twice)
    for m, fn in measures.items():
        d, hw = why.compare(A, S, lambda r, fn=fn: fn(r, why.last(r)))
        data["score"][m].update({"s": seeds_of(S, fn), "ds": r1(d, 3), "hws": r1(hw, 3)})
    # what the last hidden layer costs each optimizer on never-seen classes: accuracy read after it minus accuracy
    # read at the encoder output, run by run over the three seeds ([difference, half-width])
    def within(g, fa, fb):
        d = [fa(r, why.last(r)) - fb(r, why.last(r)) for _, r in sorted(g.items())]
        return [r1(float(np.mean(d)), 3), r1(why.interval(d), 3)]
    data["across"] = {k: {rule[0]: within(g, fs(rule, "pre_logits"), fs(rule, "gap")) for rule in ("plain", "white")}
                      for k, g in (("adamw", A), ("muon3", M), ("sgd", S))}
    # what building the model without the layer changes for each optimizer: no layer minus the normal model, seed by
    # seed.  The few-shot rows set the no-layer model against the normal one read at its last hidden layer (pl) and
    # at its encoder output (gap).
    data["drop"] = {k: {**{m: versus(base, np_, measures[m]) for m in ("val", "rare", "common")},
                        **{f"{layer}_{rule[0]}": versus(base, np_, fs(rule, "gap"), fs(rule, at))
                           for rule in ("plain", "white") for layer, at in (("pl", "pre_logits"), ("gap", "gap"))}}
                    for k, base, np_ in (("a", A, ANP), ("m", M, MNP))}
    # who trains the blocks x who trains the last hidden layer's matrix, all at learning rate 1e-3: every seed of the
    # four cells (first letter the blocks' optimizer, second the matrix's)
    grid = {"val": measures["val"], "rare": measures["rare"], "un_pl_w": measures["un_pl_w"], "un_pl_p": measures["un_pl_p"],
            "plrank": lambda r, c: c["weights"]["pre_logits.weight"]["srank"], "dim_pl": dims("pre_logits"),
            "sat": lambda r, c: 100 * c["tanh"][U]["over90"]}
    cells = {"AA": A, "AM": PLM, "MA": PLA1, "MM": M1}
    data["grid"] = {m: {k: seeds_of(g, fn) for k, g in cells.items()} for m, fn in grid.items()}
    # the mirror against the settings around it, [difference, half-width]
    data["mirror"] = {m: {"plm_adamw": versus(A, PLM, fn), "muon1_plm": versus(PLM, M1, fn), "plm3_plm": versus(PLM, PLM3, fn),
                          "plm3_adamw": versus(A, PLM3, fn), "plm3_muon3": versus(M, PLM3, fn), "pla1_adamw": versus(A, PLA1, fn)}
                      for m, fn in grid.items()}
    data["steps"] = {m: {k: r1(at_end(g, measures[m]), 3) for k, g in
                         {"adamw": A, "plm": PLM, "pla1": PLA1, "muon1": M1, "muon3": M, "pla": extra["pla"],
                          "plm3": PLM3,
                          # AdamW and Muon 3e-3 on the seeds the two control settings also have (the chained rows)
                          "adamws": {k: r for k, r in A.items() if k in PLM and k in M1},
                          "muon3s": {k: r for k, r in M.items() if k in PLM and k in M1}}.items()}
                     for m in ("un_pl_p", "un_pl_w")}
    assert len(PLM) == len(M1) == 3 and all(k in A and k in M for k in PLM) and set(PLM) == set(M1), "steps chart: shared seeds"

    # by depth at the end
    layers = any_run["probe"]["layers"]
    data["depth"] = {"layers": layers}
    for rule in ("plain", "white"):
        for k, g in {**main4, **extra}.items():
            data["depth"][f"{k}_{rule}"] = [r1(np.mean([fs(rule, L)(r, why.last(r)) for r in g.values()])) for L in layers]
        for k in ("muon3", "muon1", "sgd", ):
            data["depth"][f"d_{k}_{rule}"] = [[r1(x, 2) for x in why.compare(A, main4[k], lambda r, L=L: fs(rule, L)(r, why.last(r)))]
                                               for L in layers]
        for k in ("mlp", "attn"):
            data["depth"][f"d_{k}_{rule}"] = [[r1(x, 2) for x in why.compare(A, extra[k], lambda r, L=L: fs(rule, L)(r, why.last(r)))]
                                               for L in layers]

    # how many directions the features fill, layer by layer at the end; and every run as one point: the directions
    # its last layer fills against its few-shot accuracy there
    data["dims"] = {k: [r1(np.mean([dims(L)(r, why.last(r)) for r in g.values()])) for L in layers]
                    for k, g in {**main4, **extra}.items()}
    data["runs"] = [{"opt": r["optimizer"], "lr": r["lr"], "seed": r["seeds"][0], "variant": nice_variant(r),
                     "epochs": r["epochs"], "val": r1(100 * why.last(r)["val"]["top1"]),
                     "dim_pl": r1(dims("pre_logits")(r, why.last(r))), "dim_gap": r1(dims("gap")(r, why.last(r))),
                     "sat": r1(100 * why.last(r)["tanh"][U]["over90"]) if has_pl(r) else None,
                     **{f"{layer[:3]}_{rule[0]}": r1(fs(rule, layer)(r, why.last(r)), 2)
                        for rule in ("plain", "white") for layer in ("gap", "pre_logits")}} for r in runs]

    # the blocks: weights and their changes
    data["blocks"] = {}
    for k, g in {**main4, "mlp": extra["mlp"], "attn": extra["attn"]}.items():
        rs = list(g.values())
        end = {
            "sr_attn": mean_runs([block_stat(r, why.last(r), ATT, "srank") for r in rs]),
            "sr_mlp": mean_runs([block_stat(r, why.last(r), MLP, "srank") for r in rs]),
            "d_attn": mean_runs([block_stat(r, why.last(r), ATT, "srank", "since_first") for r in rs]),
            "d_mlp": mean_runs([block_stat(r, why.last(r), MLP, "srank", "since_first") for r in rs]),
            "moved": mean_runs([[why.moved(r, b) for b in why.block_numbers(r)] for r in rs]),
        }
        # per checkpoint (from the second on): the change since the checkpoint before, per block
        cks = [[c for c in r["probe"]["checkpoints"]] for r in rs]
        def per_ck(fams, key, sub):
            return mean_runs([[block_stat(r, c, fams, key, sub) for c in ck[1:]] for r, ck in zip(rs, cks)])
        def rel_step(fams):
            out = []
            for r, ck in zip(rs, cks):
                rows = []
                for c in ck[1:]:
                    row = []
                    for b in why.block_numbers(r):
                        ws = [c["weights"][f"blocks.{b - 1}.{f}.weight"] for f in fams]
                        row.append(float(np.sqrt(sum(w["since_last"]["fro"] ** 2 for w in ws) / sum(w["fro"] ** 2 for w in ws))))
                    rows.append(row)
                out.append(rows)
            return np.round(np.mean(np.array(out), 0), 4).tolist()
        def top_share(fams):
            """Per stretch: the share (%) of a matrix's change that lies in its strongest direction (largest
            singular value squared over the sum of all squared), mean over the blocks' matrices and the runs."""
            out = []
            for r, ck in zip(rs, cks):
                out.append([float(np.mean([100 * c["weights"][f"blocks.{b - 1}.{f}.weight"]["since_last"]["spec"] ** 2
                                           / c["weights"][f"blocks.{b - 1}.{f}.weight"]["since_last"]["fro"] ** 2
                                           for b in why.block_numbers(r) for f in fams])) for c in ck[1:]])
            return np.round(np.mean(np.array(out), 0), 2).tolist()
        end["u_top_attn"], end["u_top_mlp"] = top_share(ATT), top_share(MLP)
        end["u_sr_attn"] = per_ck(ATT, "srank", "since_last")       # [checkpoint][block]
        end["u_sr_mlp"] = per_ck(MLP, "srank", "since_last")
        end["u_size_attn"] = rel_step(ATT)
        end["u_size_mlp"] = rel_step(MLP)
        end["f_sr_attn"] = per_ck(ATT, "srank", "since_first")
        end["f_sr_mlp"] = per_ck(MLP, "srank", "since_first")
        data["blocks"][k] = end

    # the pre_logits matrix's singular values at the end (seed 0), largest first, over their median
    data["sv"] = {}
    for k, g in {**main4, "pla": extra["pla"]}.items():
        r = g.get((0, 0)) or next(iter(g.values()))
        sv = np.sort(np.array(why.last(r)["sv"]["pre_logits.weight"], dtype=float))[::-1]
        data["sv"][k] = np.round(sv / np.median(sv), 3)[::4].tolist()

    with open("icl/results/flops/flops.json") as f:
        fl = json.load(f)
    names = {"adamw": "adamw lr 0.001", "muon3": "muon lr 0.003", "muon1": "muon lr 0.001", "sgd": "sgd lr 1",
             "m70": "muon lr 0.003, 70.0 epochs"}
    data["compute"] = {}
    for k, n in names.items():
        v = fl["settings"][n]
        c = v["curve"]
        data["compute"][k] = {"eflops": round(v["eflops"], 3), "gpu_hours": round(v["gpu_hours"], 2),
                              "e": [round(x, 4) for x in c["eflops"]], "g": [round(x, 3) for x in c["gpu_hours"]],
                              "val": [r1(x, 2) for x in c["val"]], "un": [r1(x, 2) for x in c["unseen_pre_logits"]]}
    # how much compute a setting needs to reach a level of validation accuracy, on its own 90-epoch curve
    levels = [10, 20, 30, 40, 50, 60, 65, 70, data["compute"]["adamw"]["val"][-1]]   # the last: where AdamW ends
    data["reach"] = {"levels": levels}
    for k in ("adamw", "muon3", "muon1", "sgd"):
        c = data["compute"][k]
        data["reach"][k] = {x: [r1(why.at_level(c["val"], c[x], L), 3) for L in levels] for x in ("e", "g")}

    # keep the file small: bands only where the page draws them, the per-block update sizes as means over blocks
    for k, a in data["along"].items():
        for name, sv in a.items():
            if sv and name not in ("val", "un_pl_w", "un_gap_w", "un_pl_p", "un_gap_p"):
                a[name] = {"m": sv["m"]}
    for k, b in data["blocks"].items():
        for kind in ("attn", "mlp"):
            b[f"u_size_{kind}"] = [round(float(np.mean(row)), 4) for row in b[f"u_size_{kind}"]]
            del b[f"f_sr_{kind}"]
    # the tests of the five findings (results/robust/robust.py writes robust.json; run it first): every comparison as
    # {m: mean, hw: half-width, n, same: seeds with the mean's sign, p: exact permutation p, s: {seed: value}}
    with open("icl/results/robust/robust.json") as f:
        rb = json.load(f)
    s1 = rb["s1"]
    data["robust"] = {**{k: slim(v) for k, v in rb.items() if k != "s1"},
                      "s1": {"a": slim(s1["a"]), "b": slim(s1["b"]), "c": slim(s1["c"], False), "d": slim(s1["d"], False),
                             "e": slim(s1["e"], False), "f": slim(s1["f"]),
                             "g": {k: {"a": slim(v["a"]), "b": slim(v["b"]), "f": slim(v["f"])}
                                   for k, v in s1.get("g", {}).items()}}}
    # the check that old and new seeds were trained alike (results/robust/code_check.py): the page states "identical",
    # so anything else stops the build
    with open("icl/results/robust/code_check.json") as f:
        chk = json.load(f)["summary"]
    if not chk["all_same"] or chk["restart_max_loss_diff"] or chk["restart_max_val_diff"] or chk["max_lr_scale_diff"]:
        raise SystemExit(f"page_data: old and new seeds were not trained alike: {chk}")
    data["robust"]["check"] = chk
    blob = json.dumps(data, separators=(",", ":"))
    if out:
        with open(out, "w") as f:
            f.write(blob)
    src, dst = "icl/results/page.src.html", "icl/results/page.html"
    with open(src) as f:
        page = f.read()
    mark = "const DATA = null;"
    if page.count(mark) != 1:
        raise SystemExit(f"page_data: {src} must hold the line `{mark}` exactly once")
    with open(dst, "w") as f:
        f.write(folded(page).replace(mark, "const DATA = " + blob + ";"))
    print(f"page_data: {len(runs)} runs, {len(blob) // 1024} KB of numbers -> {dst} ({os.path.getsize(dst) // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
