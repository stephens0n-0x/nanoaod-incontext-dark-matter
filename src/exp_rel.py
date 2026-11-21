"""
Native-NanoAOD relational in-context learning for compressed stop -> neutralino (dark matter) signals.
Question: can an off-the-shelf relational foundation model (NVIDIA Kumo Relational) read the event as
linked object tables (event -> jets, leptons, secondary vertices, isolated tracks) and match or beat
models that need hand-engineered physics features?

Models (same few-shot protocol as exp.py: N signal + fixed weighted background context):
  rel        Kumo Relational, event table (MET only) + 4 object tables, no engineered features
  tab_flat   Kumo Tabular on naive zero-knowledge flattening (top-k objects padded into columns)
  tab_eng    Kumo Tabular on the 32 engineered physics features (project-1 setup)
  xgb_flat / xgb_eng   tuned XGBoost on the same two representations
"""
import argparse, glob, os, time, warnings
import numpy as np, pandas as pd, torch
import exp as E

warnings.filterwarnings("ignore")
TABLES = ["jets", "leps", "svs", "trks"]
FLAT_K = dict(jets=4, leps=2, svs=3, trks=2)


def load(indir, test_bkg_max, seed=777):
    evf = [f for f in sorted(glob.glob(f"{indir}/*.parquet")) if "__" not in os.path.basename(f)]
    df = pd.concat([pd.read_parquet(f) for f in evf], ignore_index=True)
    objs = {t: pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{indir}/*__{t}.parquet"))],
                         ignore_index=True) for t in TABLES}
    df["weight"] = df["weight"].clip(lower=0)
    rng = np.random.RandomState(seed)
    df["is_test"] = rng.rand(len(df)) < 0.5
    bkg, sig = df[df.process != "signal"], df[df.process == "signal"]
    btr, bte = bkg[~bkg.is_test].reset_index(drop=True), bkg[bkg.is_test]
    if len(bte) > test_bkg_max:
        w = bte.weight.values; lo, hi = 0.0, 1e12
        for _ in range(100):
            k = (lo + hi) / 2
            lo, hi = (k, hi) if np.minimum(1, k * w).sum() < test_bkg_max else (lo, k)
        p = np.minimum(1, k * w); keep = rng.rand(len(w)) < p
        bte = bte[keep].assign(weight=w[keep] / p[keep])
    return btr, bte.reset_index(drop=True), sig, objs


def flatten(ev, objs):
    """Zero-knowledge flattening: MET + top-k objects of each collection as padded columns + counts."""
    out = ev[["evt_id", "met"]].copy()
    for t, k in FLAT_K.items():
        o = objs[t][objs[t].evt_id.isin(ev.evt_id) & (objs[t]["rank"] < k)]
        cols = [c for c in o.columns if c not in ("evt_id", "rank")]
        w = o.pivot_table(index="evt_id", columns="rank", values=cols, aggfunc="first")
        w.columns = [f"{t}{int(r)}_{c}" for c, r in w.columns]
        out = out.merge(w, left_on="evt_id", right_index=True, how="left")
        n = objs[t][objs[t].evt_id.isin(ev.evt_id)].groupby("evt_id").size().rename(f"n_{t}")
        out = out.merge(n, left_on="evt_id", right_index=True, how="left")
        out[f"n_{t}"] = out[f"n_{t}"].fillna(0)
    return out.drop(columns="evt_id")


FLAT_COLS = None


def flat_X(ev, objs):
    global FLAT_COLS
    f = flatten(ev, objs)
    if FLAT_COLS is None:
        FLAT_COLS = sorted(f.columns)
    return f.reindex(columns=FLAT_COLS).astype(np.float32).values


# ----------------------------------------------------------------------------- relational model
_R = {}


def rel_tables(ev, objs, dev, offset=0):
    """Re-key events with fresh integer ids (context may contain duplicates) and build RelatedTables."""
    import sdm
    key = pd.DataFrame({"evt_id": ev.evt_id.values, "nid": np.arange(len(ev)) + offset})
    tabs = {"events": sdm.TableTensor.from_pandas(
        pd.DataFrame({"nid": key.nid.values, "met": ev.met.values.astype(np.float32)}),
        stypes={"nid": "id", "met": "numerical"}, device=dev)}
    rels = []
    for t in TABLES:
        o = key.merge(objs[t], on="evt_id", how="inner").drop(columns="evt_id")
        if len(o) == 0:  # keep schema stable
            o = pd.DataFrame({c: [] for c in ["nid"] + [c for c in objs[t].columns if c != "evt_id"]})
        st = {c: "numerical" for c in o.columns}; st["nid"] = "id"
        tabs[t] = sdm.TableTensor.from_pandas(o.astype({c: np.float32 for c in o.columns if c != "nid"}),
                                              stypes=st, device=dev)
        rels.append(dict(left_table=t, left_column="nid", right_table="events", right_column="nid"))
    rt = sdm.RelatedTables(tables=tabs, relationships=rels,
                           task_links=[dict(task_column="nid", table="events", table_column="nid")])
    x = sdm.TableTensor.from_pandas(pd.DataFrame({"nid": key.nid.values}), stypes={"nid": "id"}, device=dev)
    return x, rt


def fit_rel(ev_c, y_c, ev_q, objs, seed, n_est=4, hops=1, batch=2048, pretrained=True):
    import sdm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "m" not in _R:
        _R["m"] = sdm.models.KumoRelational(task="classification", device=dev, pretrained=pretrained)
    m = _R["m"]
    xc, rc = rel_tables(ev_c, objs, dev)
    yc = sdm.TableTensor.from_pandas(pd.DataFrame({"y": y_c.astype(int).astype(str)}), stypes={"y": "categorical"}, device=dev)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    m.fit(xc, yc["y"], rc, num_estimators=n_est, generator=g, num_hops=hops)
    out = []
    with torch.no_grad():
        for i in range(0, len(ev_q), batch):
            xq, rq = rel_tables(ev_q.iloc[i:i + batch], objs, dev, offset=10_000_000)
            p = m.predict(xq, rq).to_pandas()
            out.append(p["1"].values if "1" in p.columns else p[p.columns[-1]].values)
    m.clear() if hasattr(m, "clear") else None
    return np.concatenate(out).astype(float)


# ----------------------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True); ap.add_argument("--out", default="res_rel.csv")
    ap.add_argument("--points", default="sig_350_335,sig_400_380,sig_500_490,sig_500_420")
    ap.add_argument("--nsig", default="10,30,100,300"); ap.add_argument("--nbkg", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--test_bkg_max", type=int, default=8000)
    ap.add_argument("--test_sig_max", type=int, default=3000)
    ap.add_argument("--models", default="rel,tab_flat,tab_eng,xgb_flat,xgb_eng")
    ap.add_argument("--hops", type=int, default=1); ap.add_argument("--n_est", type=int, default=4)
    ap.add_argument("--time_budget_h", type=float, default=4.0); ap.add_argument("--random_weights", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    btr, bte, sig, objs = load(a.indir, a.test_bkg_max)
    print(f"train bkg {len(btr)}, test bkg {len(bte)} (w={bte.weight.sum():.0f}), signal {len(sig)}", flush=True)
    done = set()
    if os.path.exists(a.out):
        d = pd.read_csv(a.out); done = set(zip(d.point, d.nsig, d.seed, d.model))
    models = a.models.split(",")
    for seed in range(a.seeds):
        for pt in a.points.split(","):
            s = sig[sig["sample"] == pt]
            str_, ste = s[~s.is_test], s[s.is_test]
            ste = ste.sample(min(len(ste), a.test_sig_max), random_state=1)
            q = pd.concat([ste, bte], ignore_index=True)
            yq = np.r_[np.ones(len(ste)), np.zeros(len(bte))]; wq = np.r_[np.ones(len(ste)), bte.weight.values]
            Fq = flat_X(q, objs); Eq = q[E.FEATS].values
            for ns in map(int, a.nsig.split(",")):
                if ns > len(str_):
                    continue
                if (time.time() - t0) / 3600 > a.time_budget_h:
                    print("time budget reached", flush=True); return
                rng = np.random.RandomState(1000 * seed + ns)
                c = pd.concat([str_.sample(ns, random_state=rng), E.bkg_context(btr, a.nbkg, rng)], ignore_index=True)
                yc = np.r_[np.ones(ns), np.zeros(a.nbkg)]
                rows = []
                for mname in models:
                    if (pt, ns, seed, mname) in done:
                        continue
                    t1 = time.time()
                    if mname == "rel":
                        sc = fit_rel(c, yc, q, objs, seed, n_est=a.n_est, hops=a.hops, pretrained=not a.random_weights)
                    elif mname == "tab_flat":
                        sc = E.fit_kumo(flat_X(c, objs), yc, Fq, seed)
                    elif mname == "tab_eng":
                        sc = E.fit_kumo(c[E.FEATS].values, yc, Eq, seed)
                    elif mname == "xgb_flat":
                        sc = E.fit_xgb(flat_X(c, objs), yc, Fq, seed)
                    elif mname == "xgb_eng":
                        sc = E.fit_xgb(c[E.FEATS].values, yc, Eq, seed)
                    r = dict(point=pt, nsig=ns, seed=seed, model=mname, sec=time.time() - t1, **E.metrics(sc, yq, wq))
                    rows.append(r)
                    print(r, flush=True)
                E.append(rows, a.out)


if __name__ == "__main__":
    main()
