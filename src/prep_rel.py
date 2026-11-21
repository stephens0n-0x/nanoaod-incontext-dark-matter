"""
Relational re-extraction: same selection as prep.py, but in addition to the 32 engineered features we keep
the native NanoAOD object collections of every selected event as linked tables (jets, leptons, secondary
vertices, isolated tracks), keyed by a global event id.  Output:
  rel/{name}.parquet         event rows: evt_id, met, 32 engineered features, genWeight, weight, ...
  rel/{name}__jets.parquet   evt_id, pt, eta, dphi_met, btag, qgl
  rel/{name}__leps.parquet   evt_id, pt, eta, dphi_met, is_mu, charge, reliso, dxy, dz, sip3d
  rel/{name}__svs.parquet    evt_id, pt, eta, dphi_met, mass, dlensig, dxysig, ndof, ntracks
  rel/{name}__trks.parquet   evt_id, pt, eta, dphi_met, dxy, dz, reliso, pdgid
"""
import argparse, glob, json, os, time, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, uproot, awkward as ak
import prep
from prep import SAMPLES, LUMI_PB, dphi, files_for

BR = sorted(set(prep.BR + ["Jet_qgl", "SV_eta", "SV_phi", "SV_ntracks",
                           "IsoTrack_pt", "IsoTrack_eta", "IsoTrack_phi", "IsoTrack_dxy", "IsoTrack_dz",
                           "IsoTrack_pfRelIso03_all", "IsoTrack_pdgId"]))
CAP = dict(jets=8, leps=4, svs=6, trks=4)
TABLES = list(CAP)


def presel(a):
    trig = a.HLT_PFMET120_PFMHT120_IDTight | a.HLT_PFMETNoMu120_PFMHTNoMu120_IDTight
    flags = (a.Flag_goodVertices & a.Flag_globalSuperTightHalo2016Filter & a.Flag_HBHENoiseFilter
             & a.Flag_HBHENoiseIsoFilter & a.Flag_EcalDeadCellTriggerPrimitiveFilter & a.Flag_BadPFMuonFilter)
    mu_ok = ((a.Muon_pt > 3.5) & (np.abs(a.Muon_eta) < 2.4) & a.Muon_looseId
             & (np.abs(a.Muon_dxy) < 0.02) & (np.abs(a.Muon_dz) < 0.1)
             & ((a.Muon_pfRelIso03_all * a.Muon_pt < 5) | (a.Muon_pfRelIso03_all < 0.2)))
    el_ok = ((a.Electron_pt > 5) & (np.abs(a.Electron_eta) < 2.5) & (a.Electron_cutBased >= 1)
             & (np.abs(a.Electron_dxy) < 0.02) & (np.abs(a.Electron_dz) < 0.1)
             & ((a.Electron_pfRelIso03_all * a.Electron_pt < 5) | (a.Electron_pfRelIso03_all < 0.2)))
    nlep = ak.sum(mu_ok, axis=1) + ak.sum(el_ok, axis=1)
    jet_ok = (a.Jet_pt > 30) & (np.abs(a.Jet_eta) < 2.4) & (a.Jet_jetId >= 2)
    jpt = a.Jet_pt[jet_ok]
    ht = ak.sum(jpt, axis=1); j1 = ak.fill_none(ak.firsts(jpt), 0.0)
    pre = trig & flags & (nlep >= 1) & (a.MET_pt > 200) & (j1 > 100) & (ht > 200)
    return pre, mu_ok, el_ok


def long_table(eid, cols, cap, sort_key):
    """cols: dict name -> jagged array (same structure). Keep top-`cap` by sort_key, return long DataFrame."""
    order = ak.argsort(sort_key, ascending=False)
    cols = {k: v[order][:, :cap] for k, v in cols.items()}
    n = ak.to_numpy(ak.num(next(iter(cols.values()))))
    out = {"evt_id": np.repeat(eid, n)}
    for k, v in cols.items():
        out[k] = ak.to_numpy(ak.flatten(v)).astype(np.float32)
    out["rank"] = ak.to_numpy(ak.flatten(ak.local_index(next(iter(cols.values()))))).astype(np.float32)
    return pd.DataFrame(out)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def jet_link(o_eta, o_phi, j_eta, j_phi, j_pt, cap=8):
    """Rank (0..cap-1, pT-ordered) of the kept jet within dR<0.4 of each object, else -1."""
    order = ak.argsort(j_pt, ascending=False)
    je, jp = j_eta[order][:, :cap], j_phi[order][:, :cap]
    O = ak.zip({"e": o_eta, "p": o_phi}); Jt = ak.zip({"e": je, "p": jp})
    c = ak.cartesian({"o": O, "j": Jt}, axis=1, nested=True)
    dr = np.sqrt((c.o.e - c.j.e) ** 2 + wrap(c.o.p - c.j.p) ** 2)
    idx = ak.argmin(dr, axis=2); mn = ak.min(dr, axis=2)
    return ak.fill_none(ak.where(ak.fill_none(mn, 9.0) < 0.4, idx, -1), -1) * 1.0


def select_rel(a, prefix, offset):
    f = prep.select(a)
    if f is None:
        return None
    pre, mu_ok, el_ok = presel(a)
    a, mu_ok, el_ok = a[pre], mu_ok[pre], el_ok[pre]
    assert len(a) == len(f)
    eid = np.array([f"{prefix}:{offset + k}" for k in range(len(f))])
    f.insert(0, "evt_id", eid)
    mphi = a.MET_phi
    jok = (a.Jet_pt > 20) & (np.abs(a.Jet_eta) < 2.4) & (a.Jet_jetId >= 2)
    J = long_table(eid, dict(pt=a.Jet_pt[jok], eta=a.Jet_eta[jok], dphi_met=dphi_j(a.Jet_phi[jok], mphi),
                             phi_rel=wrap(a.Jet_phi[jok] - mphi), btag=a.Jet_btagDeepFlavB[jok], qgl=a.Jet_qgl[jok]), CAP["jets"], a.Jet_pt[jok])
    mu = dict(pt=a.Muon_pt[mu_ok], eta=a.Muon_eta[mu_ok], dphi_met=dphi_j(a.Muon_phi[mu_ok], mphi),
              is_mu=ak.ones_like(a.Muon_pt[mu_ok]), charge=a.Muon_charge[mu_ok] * 1.0,
              reliso=a.Muon_pfRelIso03_all[mu_ok], dxy=np.abs(a.Muon_dxy[mu_ok]), dz=np.abs(a.Muon_dz[mu_ok]),
              sip3d=a.Muon_sip3d[mu_ok])
    el = dict(pt=a.Electron_pt[el_ok], eta=a.Electron_eta[el_ok], dphi_met=dphi_j(a.Electron_phi[el_ok], mphi),
              is_mu=ak.zeros_like(a.Electron_pt[el_ok]), charge=a.Electron_charge[el_ok] * 1.0,
              reliso=a.Electron_pfRelIso03_all[el_ok], dxy=np.abs(a.Electron_dxy[el_ok]),
              dz=np.abs(a.Electron_dz[el_ok]), sip3d=a.Electron_sip3d[el_ok])
    mu["phi_rel"] = wrap(a.Muon_phi[mu_ok] - mphi); el["phi_rel"] = wrap(a.Electron_phi[el_ok] - mphi)
    mu["jet_rank"] = jet_link(a.Muon_eta[mu_ok], a.Muon_phi[mu_ok], a.Jet_eta[jok], a.Jet_phi[jok], a.Jet_pt[jok])
    el["jet_rank"] = jet_link(a.Electron_eta[el_ok], a.Electron_phi[el_ok], a.Jet_eta[jok], a.Jet_phi[jok], a.Jet_pt[jok])
    L = {k: ak.concatenate([mu[k], el[k]], axis=1) for k in mu}
    L = long_table(eid, L, CAP["leps"], L["pt"])
    sok = a.SV_ndof >= 1.0
    S = long_table(eid, dict(pt=a.SV_pt[sok], eta=a.SV_eta[sok], dphi_met=dphi_j(a.SV_phi[sok], mphi),
                             mass=a.SV_mass[sok], dlensig=a.SV_dlenSig[sok], dxysig=a.SV_dxySig[sok],
                             ndof=a.SV_ndof[sok], ntracks=a.SV_ntracks[sok] * 1.0,
                             phi_rel=wrap(a.SV_phi[sok] - mphi),
                             jet_rank=jet_link(a.SV_eta[sok], a.SV_phi[sok], a.Jet_eta[jok], a.Jet_phi[jok], a.Jet_pt[jok])),
                   CAP["svs"], a.SV_pt[sok])
    T = long_table(eid, dict(pt=a.IsoTrack_pt, eta=a.IsoTrack_eta, dphi_met=dphi_j(a.IsoTrack_phi, mphi),
                             dxy=np.abs(a.IsoTrack_dxy), dz=np.abs(a.IsoTrack_dz),
                             reliso=a.IsoTrack_pfRelIso03_all, pdgid=np.abs(a.IsoTrack_pdgId) * 1.0,
                             phi_rel=wrap(a.IsoTrack_phi - mphi)),
                   CAP["trks"], a.IsoTrack_pt)
    return f, dict(jets=J, leps=L, svs=S, trks=T)


def dphi_j(phi, mphi):
    d = np.abs(phi - mphi) % (2 * np.pi)
    return ak.where(d > np.pi, 2 * np.pi - d, d)


def process_file(job):
    name, i, url, outdir = job
    part = f"{outdir}/parts/{name}__{i}"
    if os.path.exists(part + ".json"):
        return
    t0 = time.time(); tmp = f"/tmp/dl/{name}__{i}.root"; os.makedirs("/tmp/dl", exist_ok=True)
    for _ in range(3):
        if os.system(f"aria2c -q -x16 -s16 -k1M --file-allocation=none -d /tmp/dl -o {name}__{i}.root '{url}'") == 0:
            break
        if os.path.exists(tmp):
            os.remove(tmp)
    t1 = time.time()
    try:
        with uproot.open(tmp) as fh:
            runs = fh["Runs"].arrays(["genEventSumw", "genEventCount"], library="np")
            sumw, cnt = float(runs["genEventSumw"].sum()), float(runs["genEventCount"].sum())
            ev, ob, off = [], {t: [] for t in TABLES}, 0
            for c in fh["Events"].iterate(BR, step_size="200 MB"):
                r = select_rel(c, f"{name}:{i}", off)
                if r is None:
                    continue
                ev.append(r[0]); off += len(r[0])
                for t in TABLES:
                    ob[t].append(r[1][t])
        df = pd.concat(ev, ignore_index=True) if ev else pd.DataFrame()
        df.to_parquet(part + ".parquet")
        for t in TABLES:
            (pd.concat(ob[t], ignore_index=True) if ob[t] else pd.DataFrame()).to_parquet(f"{part}__{t}.parquet")
        json.dump(dict(sumw=sumw, cnt=cnt, n=len(df)), open(part + ".json", "w"))
        print(f"  {name}#{i}: sel={len(df)} dl={t1-t0:.0f}s proc={time.time()-t1:.0f}s", flush=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def merge(name, outdir):
    proc, rec, xs = SAMPLES[name]
    js = sorted(glob.glob(f"{outdir}/parts/{name}__*.json"))
    meta = [(j[:-5], json.load(open(j))) for j in js]
    meta = [(p, m) for p, m in meta if True]
    if not meta or sum(m["n"] for _, m in meta) == 0:
        print(f"{name}: nothing", flush=True); return
    sumw, cnt = sum(m["sumw"] for _, m in meta), sum(m["cnt"] for _, m in meta)
    df = pd.concat([pd.read_parquet(p + ".parquet") for p, m in meta if m["n"] > 0], ignore_index=True)
    df["weight"] = xs * LUMI_PB * df.genWeight / (sumw * TOTALS[name] / cnt)
    df["sample"], df["process"] = name, proc
    if proc == "signal":
        ms, ml = map(float, name.split("_")[1:]); df["m_stop"], df["m_lsp"], df["dm"] = ms, ml, ms - ml
    df.to_parquet(f"{outdir}/{name}.parquet")
    for t in TABLES:
        pd.concat([pd.read_parquet(f"{p}__{t}.parquet") for p, m in meta if m["n"] > 0],
                  ignore_index=True).to_parquet(f"{outdir}/{name}__{t}.parquet")
    print(f"{name}: kept {len(df)} weighted={df.weight.sum():.1f}", flush=True)


TOTALS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rel"); ap.add_argument("--samples", default="all")
    ap.add_argument("--max_files", type=int, default=2); ap.add_argument("--sig_max_files", type=int, default=99)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--time_budget_h", type=float, default=10.5)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    names = list(SAMPLES) if a.samples == "all" else a.samples.split(",")
    jobs = []
    for name in names:
        urls, TOTALS[name] = files_for(SAMPLES[name][1])
        k = a.sig_max_files if SAMPLES[name][0] == "signal" else a.max_files
        jobs += [(name, i, u, a.out) for i, u in enumerate(urls[:k])]
    print(f"{len(jobs)} files", flush=True); t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        fs = [ex.submit(lambda jj: None if (time.time() - t0) / 3600 > a.time_budget_h else process_file(jj), j)
              for j in jobs]
        for f in fs:
            try:
                f.result()
            except Exception:
                print(traceback.format_exc()[-800:], flush=True)
    for name in names:
        try:
            merge(name, a.out)
        except Exception:
            print(traceback.format_exc()[-800:], flush=True)


if __name__ == "__main__":
    main()
