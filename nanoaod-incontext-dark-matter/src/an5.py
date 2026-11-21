"""v5 analysis: paired wins, geometric-mean gain ratios with bootstrap 95% CIs, per-seed spread, LOO by signal family."""
import sys
import numpy as np, pandas as pd

d = pd.read_csv(sys.argv[1])
d = d[d.error.isna() & d.gain_val.notna()]
d["kind"] = np.where(d.src.str.startswith("abl"), "abl", np.where(d["mode"] == "loo", "loo", "fs"))
KEY = ["point", "nsig", "seed"]
FAM = {"sig_300_290": "LLStop", "sig_350_335": "LLStop", "sig_400_380": "LLStop",
       "sig_500_420": "T2-4bd", "sig_500_490": "T2-4bd", "sig_525_350": "T2tt", "sig_525_438": "T2tt"}
rng = np.random.default_rng(0)


def pair(kind, a, b, B=4000):
    x = d[d.kind == kind].pivot_table(index=KEY, columns="model", values="gain_val", aggfunc="first")
    if a not in x or b not in x:
        return None
    x = x[[a, b]].dropna()
    x = x[(x[a] > 0) & (x[b] > 0)]
    if len(x) == 0:
        return None
    lr = np.log(x[a].values / x[b].values)
    # cluster bootstrap over (point, nsig) cells so the 5 seeds of a cell move together
    cells = x.reset_index()[["point", "nsig"]].astype(str).agg("|".join, axis=1).values
    u = np.unique(cells); idx = {c: np.where(cells == c)[0] for c in u}
    bs = [np.mean(np.concatenate([lr[idx[c]] for c in rng.choice(u, len(u))])) for _ in range(B)]
    lo, hi = np.exp(np.percentile(bs, [2.5, 97.5]))
    return dict(a=a, b=b, n=len(lr), wins=int((lr > 0).sum()), ratio=round(float(np.exp(lr.mean())), 3),
                ci_lo=round(float(lo), 3), ci_hi=round(float(hi), 3))


FS = [("tab_flat", "xgb_eng"), ("tab_flat", "xgb_flat"), ("tab_flat", "tab_eng"), ("tab_eng", "xgb_eng"),
      ("ticl_flat", "xgb_eng"), ("ticl_flat", "xgb_flat"), ("ticl_flat", "ticl_eng"), ("ticl_eng", "xgb_eng"),
      ("tab_flat", "ticl_flat"), ("tab_eng", "ticl_eng"),
      ("rel", "xgb_flat"), ("rel", "deepsets"), ("tab_flat", "rel"), ("rel_eng", "rel"), ("rel_eng", "tab_eng")]
AB = [(m, "rel") for m in ("rel_noLep", "rel_noSV", "rel_noTrk", "rel_noJet", "rel_h2")]
LO = [("tab_eng", "xgb_eng"), ("tab_eng", "pnn"), ("ticl_eng", "xgb_eng"), ("tab_eng", "rel"), ("tab_flat", "xgb_eng"),
      ("ticl_flat", "xgb_eng"), ("rel_eng", "rel")]
out = []
for kind, L in (("fs", FS), ("abl", AB), ("loo", LO)):
    for a, b in L:
        r = pair(kind, a, b)
        if r:
            out.append(dict(kind=kind, **r))
P = pd.DataFrame(out)
print(P.to_string(index=False))

# mean G_val per model and nsig (few-shot), with seed standard error
f = d[d.kind == "fs"]
g = f.groupby(["model", "nsig"]).gain_val
T = (g.mean().round(2).astype(str) + " ± " + (g.std() / np.sqrt(g.count())).round(2).astype(str)).unstack()
print("\nfew-shot G_val mean ± s.e. over points×seeds\n", T.to_string())

# LOO by point (family label), mean ± s.e. over seeds
l = d[d.kind == "loo"].copy(); l["family"] = l.point.map(FAM)
g = l.groupby(["family", "point", "model"]).gain_val
L5 = (g.mean().round(2).astype(str) + " ± " + (g.std() / np.sqrt(g.count())).round(2).astype(str)).unstack()
print("\nLOO G_val by point\n", L5.to_string())
P.to_csv("paired_v5.csv", index=False); T.to_csv("fs_table_v5.csv"); L5.to_csv("loo_table_v5.csv")
print("\nseeds per model:", d.groupby(["kind", "model"]).seed.nunique().to_dict())
