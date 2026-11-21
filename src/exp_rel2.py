"""
Project 2, full study: relational in-context learning on native NanoAOD object tables.

Modes
  fewshot : per mass point, N labelled signal + weighted background context (same protocol as project 1)
  loo     : leave-one-mass-point-out; signal context from the other points, (m_stop, dm) as event columns

Models
  rel        Kumo Relational: event(MET) + jets/leps/svs/trks tables, no engineered features
  rel_h2     same, 2 message-passing hops
  rel_eng    Kumo Relational with the 32 engineered features added to the event table
  rel_noSV / rel_noTrk / rel_noLep   table ablations
  deepsets   permutation-invariant network trained from scratch on the same object tables
  tab_flat / xgb_flat   naive top-k flattening; tab_eng / xgb_eng   32 engineered features

Metrics: AUC, eps_s at eps_b=1e-2/1e-3, gain (threshold optimised on the test set, as project 1) and
gain_val (threshold chosen on an independent validation half, evaluated on the test half).
"""
import argparse, os, time, warnings
import numpy as np, pandas as pd, torch
import exp as E
import exp_rel as R

warnings.filterwarnings("ignore")
PARAMS = ["m_stop", "dm"]
EXTRA = ("phi_rel", "jet_rank")
OBJS_FULL = None
COUNTS = ["n_jets", "n_leps", "n_svs", "n_trks"]


# ----------------------------------------------------------------------------- metrics
def gain_curve(s, b, wb, thr):
    out = []
    for t in thr:
        sel = b > t
        es, ebw = (s > t).mean(), wb[sel].sum() / wb.sum()
        out.append((t, es, ebw, sel.sum()))
    return out


def metrics2(score, y, w, is_val):
    """Project-1 metrics on the test half + validation-chosen working point."""
    te, va = ~is_val, is_val
    out = E.metrics(score[te], y[te], w[te])
    s_v, b_v, wb_v = score[va & (y == 1)], score[va & (y == 0)], w[va & (y == 0)]
    s_t, b_t, wb_t = score[te & (y == 1)], score[te & (y == 0)], w[te & (y == 0)]
    thr = np.unique(np.quantile(np.concatenate([s_v, b_v]), np.linspace(0.5, 0.9999, 400)))
    best, tbest = -1, None
    for t, es, ebw, nraw in gain_curve(s_v, b_v, wb_v, thr):
        if nraw >= 20 and ebw > 0 and es / np.sqrt(ebw) > best:
            best, tbest = es / np.sqrt(ebw), t
    if tbest is None:
        out["gain_val"] = np.nan
    else:
        ebt = wb_t[b_t > tbest].sum() / wb_t.sum()
        out["gain_val"] = float((s_t > tbest).mean() / np.sqrt(max(ebt, 1e-12))) if ebt > 0 else np.nan
        out["es_wp"], out["eb_wp"], out["nb_raw_wp"] = float((s_t > tbest).mean()), float(ebt), int((b_t > tbest).sum())
    return out


# ----------------------------------------------------------------------------- relational with options
def rel_tables2(ev, objs, dev, tables, ev_cols, offset=0):
    import sdm
    key = pd.DataFrame({"evt_id": ev.evt_id.values, "nid": np.arange(len(ev)) + offset})
    ed = {"nid": key.nid.values}
    for c in ev_cols:
        ed[c] = ev[c].values.astype(np.float32)
    st = {c: "numerical" for c in ev_cols}; st["nid"] = "id"
    tabs = {"events": sdm.TableTensor.from_pandas(pd.DataFrame(ed), stypes=st, device=dev)}
    rels = []
    for t in tables:
        o = key.merge(objs[t], on="evt_id", how="inner").drop(columns="evt_id")
        if len(o) == 0:
            o = pd.DataFrame({c: np.zeros(0, np.float32) for c in ["nid"] + [c for c in objs[t].columns if c != "evt_id"]})
            o["nid"] = o["nid"].astype(np.int64)
        s2 = {c: "numerical" for c in o.columns}; s2["nid"] = "id"
        tabs[t] = sdm.TableTensor.from_pandas(o.astype({c: np.float32 for c in o.columns if c != "nid"}), stypes=s2, device=dev)
        rels.append(dict(left_table=t, left_column="nid", right_table="events", right_column="nid"))
    rt = sdm.RelatedTables(tables=tabs, relationships=rels,
                           task_links=[dict(task_column="nid", table="events", table_column="nid")])
    x = sdm.TableTensor.from_pandas(pd.DataFrame({"nid": key.nid.values}), stypes={"nid": "id"}, device=dev)
    return x, rt


def fit_rel2(ev_c, y_c, ev_q, objs, seed, tables=R.TABLES, ev_cols=("met",), hops=1, n_est=4, batch=2048):
    import sdm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "m" not in R._R:
        R._R["m"] = sdm.models.KumoRelational(task="classification", device=dev, pretrained=os.environ.get("RAND") != "1")
    m = R._R["m"]
    xc, rc = rel_tables2(ev_c, objs, dev, tables, list(ev_cols))
    yc = sdm.TableTensor.from_pandas(pd.DataFrame({"y": y_c.astype(int).astype(str)}), stypes={"y": "categorical"}, device=dev)
    m.fit(xc, yc["y"], rc, num_estimators=n_est, generator=torch.Generator(device=dev).manual_seed(int(seed)), num_hops=hops)
    out = []
    with torch.no_grad():
        for i in range(0, len(ev_q), batch):
            xq, rq = rel_tables2(ev_q.iloc[i:i + batch], objs, dev, tables, list(ev_cols), offset=10_000_000)
            p = m.predict(xq, rq).to_pandas()
            out.append(p["1"].values if "1" in p.columns else p[p.columns[-1]].values)
    if hasattr(m, "clear"):
        m.clear()
    return np.concatenate(out).astype(float)


def rel_tables_link(ev, objs, dev, ev_cols, offset=0):
    """2-hop schema: events <- jets <- {svs, leps} (dR<0.4 link, null if unmatched); trks, svs, leps also -> events."""
    import sdm
    key = pd.DataFrame({"evt_id": ev.evt_id.values, "nid": np.arange(len(ev)) + offset})
    ed = {"nid": key.nid.values}
    for c in ev_cols:
        ed[c] = ev[c].values.astype(np.float32)
    st = {c: "numerical" for c in ev_cols}; st["nid"] = "id"
    tabs = {"events": sdm.TableTensor.from_pandas(pd.DataFrame(ed), stypes=st, device=dev)}
    rels = []
    for t in R.TABLES:
        o = key.merge(objs[t], on="evt_id", how="inner").drop(columns="evt_id")
        o = o.drop(columns=[c for c in ("phi_rel",) if c in o.columns])
        if t == "jets":
            o["jid"] = o["nid"] * 16 + o["rank"].astype(np.int64)
        if "jet_rank" in o.columns:
            jr = o.pop("jet_rank").values
            o["jid"] = np.where(jr >= 0, o["nid"] * 16 + jr, np.nan)
        ids = [c for c in ("nid", "jid") if c in o.columns]
        s2 = {c: "numerical" for c in o.columns}
        for c in ids:
            s2[c] = "id"
        o = o.astype({c: np.float32 for c in o.columns if c not in ids})
        tabs[t] = sdm.TableTensor.from_pandas(o, stypes=s2, device=dev)
        rels.append(dict(left_table=t, left_column="nid", right_table="events", right_column="nid"))
        if t in ("svs", "leps"):
            rels.append(dict(left_table=t, left_column="jid", right_table="jets", right_column="jid"))
    rt = sdm.RelatedTables(tables=tabs, relationships=rels,
                           task_links=[dict(task_column="nid", table="events", table_column="nid")])
    x = sdm.TableTensor.from_pandas(pd.DataFrame({"nid": key.nid.values}), stypes={"nid": "id"}, device=dev)
    return x, rt


def fit_rel_link(ev_c, y_c, ev_q, seed, ev_cols, hops=2, n_est=8, batch=2048):
    import sdm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "m" not in R._R:
        R._R["m"] = sdm.models.KumoRelational(task="classification", device=dev, pretrained=os.environ.get("RAND") != "1")
    m = R._R["m"]
    xc, rc = rel_tables_link(ev_c, OBJS_FULL, dev, list(ev_cols))
    yc = sdm.TableTensor.from_pandas(pd.DataFrame({"y": y_c.astype(int).astype(str)}), stypes={"y": "categorical"}, device=dev)
    m.fit(xc, yc["y"], rc, num_estimators=n_est, generator=torch.Generator(device=dev).manual_seed(int(seed)), num_hops=hops)
    out = []
    with torch.no_grad():
        for i in range(0, len(ev_q), batch):
            xq, rq = rel_tables_link(ev_q.iloc[i:i + batch], OBJS_FULL, dev, list(ev_cols), offset=10_000_000)
            out.append(m.predict(xq, rq).to_pandas()["1"].values)
    if hasattr(m, "clear"):
        m.clear()
    return np.concatenate(out).astype(float)


# ----------------------------------------------------------------------------- DeepSets baseline
OBJ_FEATS = None


def set_tensors(ev, objs, ev_cols, stats=None):
    """Padded per-table tensors [n_ev, K, F] + masks, log-pt, standardised with context stats."""
    global OBJ_FEATS
    idx = pd.Series(np.arange(len(ev)), index=ev.evt_id.values)
    X, M, new_stats = {}, {}, {}
    for t in R.TABLES:
        K = {"jets": 8, "leps": 4, "svs": 6, "trks": 4}[t]
        o = objs[t][objs[t].evt_id.isin(idx.index)]
        cols = [c for c in o.columns if c not in ("evt_id", "rank")]
        a = np.zeros((len(ev), K, len(cols)), np.float32); m = np.zeros((len(ev), K), np.float32)
        # duplicates in ev (resampled context) -> map each row; use first match per evt then broadcast
        pos = {}
        for r_, e in enumerate(ev.evt_id.values):
            pos.setdefault(e, []).append(r_)
        v = o[cols].values.astype(np.float32)
        if "pt" in cols:
            v[:, cols.index("pt")] = np.log1p(v[:, cols.index("pt")])
        rk = o["rank"].values.astype(int)
        for e, r_k, row in zip(o.evt_id.values, rk, v):
            if r_k < K:
                for r_ in pos[e]:
                    a[r_, r_k] = row; m[r_, r_k] = 1
        if stats is None:
            flat = a[m > 0]
            mu, sd = (flat.mean(0), flat.std(0) + 1e-6) if len(flat) else (np.zeros(len(cols)), np.ones(len(cols)))
            new_stats[t] = (mu, sd)
        else:
            mu, sd = stats[t]
        a = np.where(m[..., None] > 0, (a - mu) / sd, 0.0).astype(np.float32)
        X[t], M[t] = a, m
    e = np.log1p(ev[list(ev_cols)].values.astype(np.float32))
    if stats is None:
        sde = e.std(0); new_stats["ev"] = (e.mean(0), np.where(sde < 1e-3, 1.0, sde))
    mu, sd = (new_stats if stats is None else stats)["ev"]
    X["ev"] = ((e - mu) / sd).astype(np.float32)
    return X, M, (new_stats if stats is None else stats)


class DeepSets(torch.nn.Module):
    def __init__(self, dims, d_ev, h=64):
        super().__init__()
        self.phi = torch.nn.ModuleDict({t: torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(),
                                                               torch.nn.Linear(h, h), torch.nn.ReLU()) for t, d in dims.items()})
        self.rho = torch.nn.Sequential(torch.nn.Linear(2 * h * len(dims) + d_ev, h), torch.nn.ReLU(),
                                       torch.nn.Dropout(0.2), torch.nn.Linear(h, 1))

    def forward(self, X, M):
        z = []
        for t in self.phi:
            hh = self.phi[t](X[t]) * M[t][..., None]
            n = M[t].sum(1, keepdim=True).clamp(min=1)
            z += [hh.sum(1) / 8.0, hh.sum(1) / n]
        return self.rho(torch.cat(z + [X["ev"]], 1)).squeeze(1)


class SetTransformer(torch.nn.Module):
    """Objects of all tables -> shared token space (+ table embedding), 2 self-attention blocks, pooled."""
    def __init__(self, dims, d_ev, h=64, heads=4, layers=2):
        super().__init__()
        self.inp = torch.nn.ModuleDict({t: torch.nn.Linear(d, h) for t, d in dims.items()})
        self.temb = torch.nn.Parameter(torch.randn(len(dims), h) * 0.02)
        self.ev = torch.nn.Linear(d_ev, h)
        enc = torch.nn.TransformerEncoderLayer(h, heads, 2 * h, dropout=0.1, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(enc, layers)
        self.out = torch.nn.Sequential(torch.nn.Linear(2 * h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))

    def forward(self, X, M):
        toks, masks = [self.ev(X["ev"])[:, None]], [torch.ones(X["ev"].shape[0], 1, device=X["ev"].device)]
        for i, t in enumerate(self.inp):
            toks.append(self.inp[t](X[t]) + self.temb[i]); masks.append(M[t])
        z = torch.cat(toks, 1); m = torch.cat(masks, 1)
        z = self.enc(z, src_key_padding_mask=(m == 0))
        pooled = (z * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        return self.out(torch.cat([pooled, z[:, 0]], 1)).squeeze(1)


def fit_settf(ev_c, y_c, ev_q, objs, seed, ev_cols=("met",), epochs=120):
    return fit_deepsets(ev_c, y_c, ev_q, objs, seed, ev_cols, epochs, arch=SetTransformer)


def fit_pnn(Xc, y_c, Xq, seed, epochs=200):
    """Parameterised MLP (Baldi et al. 2016) on engineered features + masses; NaNs imputed with context medians."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    med = np.nanmedian(Xc, 0); med = np.where(np.isnan(med), 0, med)
    f = lambda X: np.where(np.isnan(X), med, X)
    Xc, Xq = f(Xc), f(Xq); mu, sd = Xc.mean(0), Xc.std(0); sd = np.where(sd < 1e-3, 1.0, sd)
    Tc = torch.tensor((Xc - mu) / sd, dtype=torch.float32, device=dev)
    Tq = torch.tensor((Xq - mu) / sd, dtype=torch.float32, device=dev)
    y = torch.tensor(y_c, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(Tc.shape[1], 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y_c == 0).sum() / max((y_c == 1).sum(), 1), device=dev))
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(y), device=dev)
        for i in range(0, len(y), 512):
            b = perm[i:i + 512]; opt.zero_grad(); lossf(net(Tc[b]).squeeze(1), y[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Tq).squeeze(1)).cpu().numpy().astype(float)


def fit_deepsets(ev_c, y_c, ev_q, objs, seed, ev_cols=("met",), epochs=150, arch=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    Xc, Mc, st = set_tensors(ev_c, objs, ev_cols)
    Xq, Mq, _ = set_tensors(ev_q, objs, ev_cols, st)
    T = lambda D: {k: torch.tensor(v, device=dev) for k, v in D.items()}
    Xc, Mc, Xq, Mq = T(Xc), T(Mc), T(Xq), T(Mq)
    y = torch.tensor(y_c, dtype=torch.float32, device=dev)
    net = (arch or DeepSets)({t: Xc[t].shape[2] for t in R.TABLES}, Xc["ev"].shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    pw = torch.tensor((y_c == 0).sum() / max((y_c == 1).sum(), 1), device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    n = len(y_c)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            l = lossf(net({k: v[b] for k, v in Xc.items()}, {k: v[b] for k, v in Mc.items()}), y[b])
            l.backward(); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(ev_q), 8192):
            out.append(torch.sigmoid(net({k: v[i:i + 8192] for k, v in Xq.items()}, {k: v[i:i + 8192] for k, v in Mq.items()})).cpu().numpy())
    return np.concatenate(out).astype(float)



def fit_ticl(Xc, y, Xq, seed):
    """TabICLv2 (Qu et al. 2026, arXiv:2602.11139), default checkpoint and settings; NaN = missing object."""
    from tabicl import TabICLClassifier
    import torch
    m = TabICLClassifier(random_state=int(seed), device="cuda" if torch.cuda.is_available() else "cpu")
    m.fit(np.asarray(Xc, dtype=np.float32), np.asarray(y).astype(int))
    p = m.predict_proba(np.asarray(Xq, dtype=np.float32))
    return p[:, list(m.classes_).index(1)].astype(float)



TPFN_VER = None


def fit_tpfn(Xc, y, Xq, seed):
    """TabPFN (Prior Labs): v2.5 if its licence is accepted for TABPFN_TOKEN, else the original v2."""
    global TPFN_VER
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xc, Xq = np.asarray(Xc, dtype=np.float32), np.asarray(Xq, dtype=np.float32)
    for v in ([TPFN_VER] if TPFN_VER else [ModelVersion.V2_5, ModelVersion.V2]):
        try:
            m = TabPFNClassifier.create_default_for_version(v, device=dev, random_state=int(seed), ignore_pretraining_limits=True)
            m.fit(Xc, np.asarray(y).astype(int))
            p = np.concatenate([m.predict_proba(Xq[i:i + 4000]) for i in range(0, len(Xq), 4000)])
            if TPFN_VER is None:
                TPFN_VER = v; print("TabPFN version used:", v, flush=True)
            return p[:, list(m.classes_).index(1)].astype(float)
        except Exception as e:
            print("TabPFN", v, "failed:", repr(e)[:300], flush=True)
            if TPFN_VER:
                raise
    raise RuntimeError("no TabPFN version available")


# ----------------------------------------------------------------------------- driver
COLS = ["mode", "point", "nsig", "seed", "model", "sec", "auc", "effs_at_0.01", "effs_at_0.001", "gain", "gain_val", "es_wp", "eb_wp", "nb_raw_wp", "error"]


def append(rows, out):
    if rows:
        pd.DataFrame(rows).reindex(columns=COLS).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
def run_model(name, c, yc, q, objs, seed, Fc, Fq, a, extra=()):
    ev_base = ["met"] + list(extra)
    if name == "rel":
        return fit_rel2(c, yc, q, objs, seed, ev_cols=ev_base, n_est=a.n_est)
    if name == "rel_h2":
        return fit_rel2(c, yc, q, objs, seed, ev_cols=ev_base, hops=2, n_est=a.n_est)
    if name == "rel_eng":
        return fit_rel2(c, yc, q, objs, seed, ev_cols=sorted(set(E.FEATS + list(extra))), n_est=a.n_est)
    if name.startswith("rel_no"):
        drop = {"rel_noSV": "svs", "rel_noTrk": "trks", "rel_noLep": "leps", "rel_noJet": "jets"}[name]
        return fit_rel2(c, yc, q, objs, seed, tables=[t for t in R.TABLES if t != drop], ev_cols=ev_base, n_est=a.n_est)
    if name == "rel8c":
        return fit_rel2(c, yc, q, objs, seed, ev_cols=ev_base + COUNTS, n_est=8)
    if name == "rel_link":
        return fit_rel_link(c, yc, q, seed, ev_cols=ev_base + COUNTS, hops=2, n_est=8)
    if name == "rel_link1":
        return fit_rel_link(c, yc, q, seed, ev_cols=ev_base + COUNTS, hops=1, n_est=8)
    if name == "settf":
        return fit_settf(c, yc, q, objs, seed, ev_cols=ev_base)
    if name == "tab_flat_s":
        return E.fit_kumo(Fc, yc, Fq, seed, size="small")
    if name == "pnn":
        feats = E.FEATS + list(extra)
        return fit_pnn(c[feats].values.astype(np.float64), yc, q[feats].values.astype(np.float64), seed)
    if name == "deepsets":
        return fit_deepsets(c, yc, q, objs, seed, ev_cols=ev_base)
    if name == "tpfn_flat":
        return fit_tpfn(Fc, yc, Fq, seed)
    if name == "tpfn_eng":
        return fit_tpfn(c[feats].values, yc, q[feats].values, seed)
    if name == "ticl_flat":
        return fit_ticl(Fc, yc, Fq, seed)
    if name == "tab_flat":
        return E.fit_kumo(Fc, yc, Fq, seed)
    if name == "xgb_flat":
        return E.fit_xgb(Fc, yc, Fq, seed)
    feats = E.FEATS + list(extra)
    if name == "tab_eng":
        return E.fit_kumo(c[feats].values, yc, q[feats].values, seed)
    if name == "ticl_eng":
        return fit_ticl(c[feats].values, yc, q[feats].values, seed)
    if name == "xgb_eng":
        return E.fit_xgb(c[feats].values, yc, q[feats].values, seed)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True); ap.add_argument("--out", default="res2.csv")
    ap.add_argument("--mode", default="fewshot")
    ap.add_argument("--points", default="sig_300_290,sig_350_335,sig_400_380,sig_500_490,sig_500_420,sig_525_438,sig_525_350")
    ap.add_argument("--nsig", default="10,30,100,300,1000"); ap.add_argument("--nbkg", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--test_bkg_max", type=int, default=12000); ap.add_argument("--test_sig_max", type=int, default=3000)
    ap.add_argument("--models", default="rel,rel_eng,deepsets,tab_flat,xgb_flat,tab_eng,xgb_eng")
    ap.add_argument("--loo_per_point", type=int, default=300)
    ap.add_argument("--n_est", type=int, default=4); ap.add_argument("--time_budget_h", type=float, default=8.0)
    a = ap.parse_args()
    t0 = time.time()
    btr, bte, sig, objs = R.load(a.indir, a.test_bkg_max)
    global OBJS_FULL
    OBJS_FULL = objs
    objs = {t: d.drop(columns=[c for c in EXTRA if c in d.columns]) for t, d in objs.items()}
    cnt = {t: objs[t].groupby("evt_id").size() for t in R.TABLES}
    def add_counts(df):
        for t in R.TABLES:
            df[f"n_{t}"] = df.evt_id.map(cnt[t]).fillna(0).astype(np.float32).values
        return df
    btr, bte, sig = add_counts(btr), add_counts(bte), add_counts(sig)
    rng0 = np.random.RandomState(4242)
    bte = bte.assign(is_val=rng0.rand(len(bte)) < 0.5)
    print(f"train bkg {len(btr)}, test bkg {len(bte)} (w={bte.weight.sum():.0f}), signal {len(sig)}", flush=True)
    import json
    LUMI = 16800.0  # pb^-1; signal stored with a 1 pb placeholder cross section, so sum(w)/L = selection acceptance
    Y = dict(lumi_pb=LUMI, b_total=float(btr.weight.sum() + bte.weight.sum()),
             b_by_process=bte.groupby("process").weight.sum().mul(2).round(2).to_dict(),
             acc_sel={p: float(g.weight.sum() / LUMI) for p, g in sig.groupby("sample")})
    print("YIELDS", json.dumps(Y), flush=True); json.dump(Y, open(a.out.replace(".csv", "_yields.json"), "w"))
    done = set()
    if os.path.exists(a.out):
        d = pd.read_csv(a.out); done = set(zip(d["mode"], d.point, d.nsig, d.seed, d.model))
    models = a.models.split(",")
    pts = a.points.split(",")
    for seed in range(a.seed0, a.seed0 + a.seeds):
        for pt in pts:
            s = sig[sig["sample"] == pt]
            str_, ste = s[~s.is_test], s[s.is_test]
            ste = ste.sample(min(len(ste), a.test_sig_max), random_state=1)
            ste = ste.assign(is_val=np.random.RandomState(7).rand(len(ste)) < 0.5)
            q = pd.concat([ste, bte], ignore_index=True)
            yq = np.r_[np.ones(len(ste)), np.zeros(len(bte))]; wq = np.r_[np.ones(len(ste)), bte.weight.values]
            vq = q.is_val.values.astype(bool)
            if a.mode == "loo":
                ms, dm = float(s.m_stop.iloc[0]), float(s.dm.iloc[0])
                q = q.assign(m_stop=ms, dm=dm)
                sizes = [-1]
            else:
                sizes = [n for n in map(int, a.nsig.split(",")) if n <= len(str_)]
            Fq = R.flat_X(q, objs)
            if a.mode == "loo":
                Fq = np.c_[Fq, q[PARAMS].values]
            for ns in sizes:
                if (time.time() - t0) / 3600 > a.time_budget_h:
                    print("time budget reached", flush=True); return
                rng = np.random.RandomState(1000 * seed + max(ns, 0) + (7 if a.mode == "loo" else 0))
                if a.mode == "loo":
                    oth = sig[(sig["sample"] != pt) & ~sig.is_test]
                    cs = pd.concat([g.sample(min(len(g), a.loo_per_point), random_state=rng) for _, g in oth.groupby("sample")])
                    cb = E.bkg_context(btr, a.nbkg, rng).copy()
                    pick = cs[PARAMS].drop_duplicates().values
                    cb[PARAMS] = pick[rng.randint(len(pick), size=len(cb))]
                    c = pd.concat([cs, cb], ignore_index=True)
                    yc = np.r_[np.ones(len(cs)), np.zeros(len(cb))]
                    extra = PARAMS
                else:
                    c = pd.concat([str_.sample(ns, random_state=rng), E.bkg_context(btr, a.nbkg, rng)], ignore_index=True)
                    yc = np.r_[np.ones(ns), np.zeros(a.nbkg)]
                    extra = ()
                Fc = R.flat_X(c, objs)
                if a.mode == "loo":
                    Fc = np.c_[Fc, c[PARAMS].values]
                rows = []
                for mname in models:
                    if (a.mode, pt, ns, seed, mname) in done:
                        continue
                    t1 = time.time()
                    try:
                        sc = run_model(mname, c, yc, q, objs, seed, Fc, Fq, a, extra)
                        r = dict(mode=a.mode, point=pt, nsig=ns, seed=seed, model=mname, sec=time.time() - t1,
                                 **metrics2(sc, yq, wq, vq))
                    except Exception as ex:  # keep going; record the failure
                        r = dict(mode=a.mode, point=pt, nsig=ns, seed=seed, model=mname, sec=time.time() - t1,
                                 error=str(ex)[:200])
                    rows.append(r)
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                append(rows, a.out)


if __name__ == "__main__":
    main()
