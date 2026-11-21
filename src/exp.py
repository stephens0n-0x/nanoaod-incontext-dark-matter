"""
Few-shot and cross-mass-point experiments for compressed stop -> neutralino (dark matter) signals
on CMS 2016 Open Data. Compares NVIDIA Kumo Tabular (in-context, no training) with tuned XGBoost,
logistic regression and fixed CMS-style cuts. Results are appended row by row (resumable).

fewshot : per mass point, N labelled signal events + fixed background context.
loo     : leave-one-mass-point-out, parameterised by (m_stop, dm); the target point's own
          simulation is never used for training.
"""
import argparse, glob, os, time, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")
FEATS = ["lep_pt", "lep_eta", "lep_is_mu", "lep_charge", "lep_reliso", "lep_dxy", "lep_dz", "lep_sip3d",
         "n_lep", "met", "mt", "dphi_lep_met", "ht", "n_jet", "n_bjet_m", "n_bjet_l", "jet1_pt", "jet1_eta",
         "jet1_btag", "jet2_pt", "jet2_btag", "dphi_j1_met", "dphi_j1_j2", "n_softsv", "sv1_pt",
         "sv1_dlensig", "sv1_mass", "n_tau", "n_isotrack", "ct1", "ct2", "lep_pt_over_met"]
PARAMS = ["m_stop", "dm"]


# ----------------------------------------------------------------------------- data
def load(indir, test_bkg_max, seed=777):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{indir}/*.parquet"))], ignore_index=True)
    df["weight"] = df["weight"].clip(lower=0)
    rng = np.random.RandomState(seed)
    df["is_test"] = rng.rand(len(df)) < 0.5
    bkg = df[df.process != "signal"]
    sig = df[df.process == "signal"]
    btr, bte = bkg[~bkg.is_test].reset_index(drop=True), bkg[bkg.is_test]
    if len(bte) > test_bkg_max:  # importance thinning: keep heavy-weight events, reweight by 1/p
        w = bte.weight.values
        lo, hi = 0.0, 1e12
        for _ in range(100):
            k = (lo + hi) / 2
            lo, hi = (k, hi) if np.minimum(1, k * w).sum() < test_bkg_max else (lo, k)
        p = np.minimum(1, k * w)
        keep = rng.rand(len(w)) < p
        bte = bte[keep].assign(weight=w[keep] / p[keep])
    return btr, bte.reset_index(drop=True), sig


def bkg_context(btr, n, rng):
    p = btr.weight.values / btr.weight.values.sum()
    return btr.iloc[rng.choice(len(btr), n, replace=True, p=p)]


# ----------------------------------------------------------------------------- metrics
def metrics(score, y, w):
    """Signal unweighted (single mass point), background weighted by cross section."""
    out = dict(auc=roc_auc_score(y, score, sample_weight=np.where(y == 1, 1.0, w)))
    s, b, wb = score[y == 1], score[y == 0], w[y == 0]
    for eb in (1e-2, 1e-3):
        o = np.argsort(-b); cw = np.cumsum(wb[o]) / wb.sum()
        k = np.searchsorted(cw, eb)
        thr = b[o][min(k, len(o) - 1)]
        out[f"effs_at_{eb:g}"] = float((s > thr).mean())
    # best normalisation-free gain eps_s / sqrt(eps_b), with >= 20 raw background events selected
    thr = np.unique(np.quantile(np.concatenate([s, b]), np.linspace(0.5, 0.9999, 400)))
    best = 0.0
    for t in thr:
        sel = b > t
        if sel.sum() < 20:
            continue
        es, eb_ = (s > t).mean(), wb[sel].sum() / wb.sum()
        if eb_ > 0:
            best = max(best, es / np.sqrt(eb_))
    out["gain"] = best
    return out


# ----------------------------------------------------------------------------- models
def gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


GRID = [dict(max_depth=d, n_estimators=n, learning_rate=0.1, min_child_weight=c, subsample=0.8,
             colsample_bytree=0.8) for d in (2, 3, 5) for n in (50, 200) for c in (1, 5)]


def fit_xgb(X, y, Xq, seed, tune=True):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    base = dict(random_state=seed, tree_method="hist", verbosity=0, scale_pos_weight=spw,
                device="cuda" if gpu() else "cpu")
    best_p, k = GRID[0], int(min(3, (y == 1).sum()))
    if tune and k >= 2:
        best = -1
        skf = StratifiedKFold(k, shuffle=True, random_state=seed)
        for p in GRID:
            sc = [roc_auc_score(y[va], xgb.XGBClassifier(**base, **p).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1])
                  for tr, va in skf.split(X, y)]
            if np.mean(sc) > best:
                best, best_p = np.mean(sc), p
    m = xgb.XGBClassifier(**base, **best_p).fit(X, y)
    return m.predict_proba(Xq)[:, 1]


def fit_logreg(X, y, Xq, seed):
    m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                      LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))
    return m.fit(X, y).predict_proba(Xq)[:, 1]


_K = {}


def fit_kumo(X, y, Xq, seed, size="large", n_est=8, batch=8192):
    import torch, sdm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if size not in _K:
        _K[size] = sdm.models.KumoTabular(task="classification", size=size, device=dev)
    m = _K[size]
    cols = [f"f{i}" for i in range(X.shape[1])]; st = {c: "numerical" for c in cols}
    m.fit(x=sdm.TableTensor.from_pandas(pd.DataFrame(X, columns=cols), stypes=st, device=dev),
          y=sdm.TableTensor.from_pandas(pd.DataFrame({"y": y.astype(int).astype(str)}), stypes={"y": "categorical"}, device=dev)["y"],
          num_estimators=n_est, generator=torch.Generator(device=dev).manual_seed(seed))
    out = []
    with torch.no_grad():
        for i in range(0, len(Xq), batch):
            p = m.predict(sdm.TableTensor.from_pandas(pd.DataFrame(Xq[i:i + batch], columns=cols),
                                                      stypes=st, device=dev)).to_pandas()
            out.append(p["1" if "1" in p.columns else p.columns[-1]].to_numpy(float))
    m.clear()
    return np.concatenate(out)


def cms_cuts(df):
    """Fixed, physics-motivated soft-lepton selection (in the spirit of CMS-SUS-17-005); score = MET."""
    ok = (df.lep_pt < 30) & (df.met > 280) & (df.ct1 > 300) & (df.jet1_pt > 110) & (df.n_jet <= 2) \
         & (df.dphi_j1_j2.fillna(0) < 2.5) & (df.n_tau == 0)
    return np.where(ok, df.met.values, -1.0)


# ----------------------------------------------------------------------------- driver
def append(rows, out):
    pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", default="res.csv")
    ap.add_argument("--mode", default="fewshot", choices=["fewshot", "loo"])
    ap.add_argument("--points", default="all")
    ap.add_argument("--nsig", default="10,30,100,300,1000")
    ap.add_argument("--nbkg", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--test_bkg_max", type=int, default=15000)
    ap.add_argument("--models", default="kumo,xgb_tuned,xgb_fullbkg,logreg,cuts")
    ap.add_argument("--loo_per_point", type=int, default=300)
    ap.add_argument("--time_budget_h", type=float, default=99)
    a = ap.parse_args()
    t0 = time.time()
    btr, bte, sig = load(a.indir, a.test_bkg_max)
    points = sorted(sig["sample"].unique()) if a.points == "all" else a.points.split(",")
    print(f"bkg train {len(btr)} test {len(bte)} | signal per point:",
          sig.groupby("sample").size().to_dict(), flush=True)
    done = set()
    if os.path.exists(a.out):
        r = pd.read_csv(a.out); done = set(zip(r.mode, r.point, r.n_sig, r.seed, r.model))
    models = a.models.split(",")

    for seed in range(a.seeds):
        rng = np.random.RandomState(seed)
        for pt in points:
            ste = sig[(sig["sample"] == pt) & sig.is_test]
            test = pd.concat([ste, bte], ignore_index=True)
            yt = (test.process == "signal").astype(int).values
            wt = test.weight.values
            if a.mode == "fewshot":
                pool = sig[(sig["sample"] == pt) & ~sig.is_test]
                configs = [(int(n), pool.sample(int(n), random_state=seed)) for n in a.nsig.split(",")
                           if int(n) <= len(pool)]
            else:  # leave-one-point-out, parameterised
                others = sig[(sig["sample"] != pt) & ~sig.is_test]
                tr_sig = others.groupby("sample", group_keys=False).apply(
                    lambda g: g.sample(min(a.loo_per_point, len(g)), random_state=seed))
                configs = [(-1, tr_sig)]
            for n, tr_sig in configs:
                bc = bkg_context(btr, a.nbkg, rng)
                feats = FEATS + (PARAMS if a.mode == "loo" else [])
                if a.mode == "loo":  # background gets masses drawn from the training signal points
                    mp = tr_sig[PARAMS].drop_duplicates().values
                    bc = bc.copy(); bc[PARAMS] = mp[rng.randint(len(mp), size=len(bc))]
                    test = test.copy(); test[PARAMS] = [float(pt.split("_")[1]), float(pt.split("_")[1]) - float(pt.split("_")[2])]
                X = np.vstack([tr_sig[feats].values, bc[feats].values]).astype(np.float32)
                y = np.r_[np.ones(len(tr_sig), int), np.zeros(len(bc), int)]
                Xq = test[feats].values.astype(np.float32)
                for mname in models:
                    if (a.mode, pt, n, seed, mname) in done:
                        continue
                    if (time.time() - t0) / 3600 > a.time_budget_h:
                        print("time budget reached", flush=True); return
                    t1 = time.time()
                    if mname == "kumo":
                        sc = fit_kumo(X, y, Xq, seed)
                    elif mname == "xgb_tuned":
                        sc = fit_xgb(X, y, Xq, seed)
                    elif mname == "xgb_fullbkg":  # XGBoost may use the whole weighted background pool
                        bb = btr.copy()
                        if a.mode == "loo":
                            bb[PARAMS] = mp[rng.randint(len(mp), size=len(bb))]
                        Xf = np.vstack([tr_sig[feats].values, bb[feats].values]).astype(np.float32)
                        yf = np.r_[np.ones(len(tr_sig), int), np.zeros(len(bb), int)]
                        wf = np.r_[np.full(len(tr_sig), bb.weight.sum() / len(tr_sig)), bb.weight.values]
                        m = xgb.XGBClassifier(random_state=seed, tree_method="hist", verbosity=0, max_depth=3,
                                              n_estimators=200, learning_rate=0.1, subsample=0.8,
                                              device="cuda" if gpu() else "cpu").fit(Xf, yf, sample_weight=wf)
                        sc = m.predict_proba(Xq)[:, 1]
                    elif mname == "logreg":
                        sc = fit_logreg(X, y, Xq, seed)
                    elif mname == "cuts":
                        if seed > 0 or n not in (-1, int(a.nsig.split(",")[0])):
                            continue  # deterministic, evaluate once per point
                        sc = cms_cuts(test)
                    m_ = metrics(sc, yt, wt)
                    append([dict(mode=a.mode, point=pt, n_sig=n, seed=seed, model=mname,
                                 n_sig_test=int(yt.sum()), seconds=time.time() - t1, **m_)], a.out)
                    print(f"[{(time.time()-t0)/60:6.1f}m] {a.mode} {pt} n={n} s={seed} {mname:11s} "
                          f"auc={m_['auc']:.3f} gain={m_['gain']:.2f} effs@1e-3={m_['effs_at_0.001']:.3f}",
                          flush=True)


if __name__ == "__main__":
    main()
