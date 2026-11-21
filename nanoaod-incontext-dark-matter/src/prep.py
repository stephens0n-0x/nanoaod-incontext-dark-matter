"""
Compressed stop -> neutralino (dark-matter-motivated) event selection on CMS 2016 Open Data (NanoAOD).
Reads files straight from CERN EOS over HTTPS, applies a soft-lepton + MET + ISR-jet preselection
(in the spirit of CMS-SUS-17-005), builds per-event features and writes one parquet per sample.
"""
import argparse, glob, json, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests, uproot, awkward as ak

LUMI_PB = 16800.0  # UL2016 post-VFP (Run2016F-late, G, H), approx.

# process, record id, cross section [pb] (LO/NLO values commonly used in CMS 2016 analyses; approximate)
SAMPLES = {
    # ---- signal: compressed stop pair -> 4-body / long-lived stop; xsec irrelevant for normalisation-free metrics
    "sig_500_420": ("signal", 63443, 1.0), "sig_500_490": ("signal", 63445, 1.0),
    "sig_300_290": ("signal", 63451, 1.0), "sig_350_335": ("signal", 63453, 1.0),
    "sig_400_380": ("signal", 63455, 1.0), "sig_525_350": ("signal", 63460, 1.0),
    "sig_525_438": ("signal", 63462, 1.0),
    # ---- backgrounds
    "WJ_200_400": ("wjets", 69725, 359.7), "WJ_400_600": ("wjets", 69729, 48.91),
    "WJ_600_800": ("wjets", 69731, 12.05), "WJ_800_1200": ("wjets", 69735, 5.501),
    "WJ_1200_2500": ("wjets", 69723, 1.329), "WJ_2500_inf": ("wjets", 69727, 0.03216),
    "TT_semilep": ("ttbar", 67993, 365.34), "TT_dilep": ("ttbar", 67801, 88.29),
    "ZNN_200_400": ("znunu", 74898, 77.67),
    "ZNN_400_600": ("znunu", 74902, 10.73), "ZNN_600_800": ("znunu", 74904, 2.559),
    "ZNN_800_1200": ("znunu", 74906, 1.1796), "ZNN_1200_2500": ("znunu", 74896, 0.28833),
    "ZNN_2500_inf": ("znunu", 74900, 0.006945),
    "WZ_3lnu": ("diboson", 72750, 4.43),
    # single top (NLO, LHC Top WG values commonly used in CMS 2016 analyses; approximate)
    "ST_tW_top": ("singletop", 64881, 35.85), "ST_tW_antitop": ("singletop", 64825, 35.85),
    "ST_t_top": ("singletop", 64759, 136.02), "ST_t_antitop": ("singletop", 64659, 80.95),
}

BR = ["genWeight", "MET_pt", "MET_phi",
      "HLT_PFMET120_PFMHT120_IDTight", "HLT_PFMETNoMu120_PFMHTNoMu120_IDTight",
      "Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter",
      "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter",
      "Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge", "Muon_looseId", "Muon_pfRelIso03_all",
      "Muon_dxy", "Muon_dz", "Muon_sip3d",
      "Electron_pt", "Electron_eta", "Electron_phi", "Electron_charge", "Electron_cutBased",
      "Electron_pfRelIso03_all", "Electron_dxy", "Electron_dz", "Electron_sip3d",
      "Jet_pt", "Jet_eta", "Jet_phi", "Jet_jetId", "Jet_btagDeepFlavB",
      "SV_pt", "SV_dlenSig", "SV_ndof", "SV_mass", "SV_dxySig",
      "Tau_pt", "nIsoTrack"]
BTAG_M = 0.2489  # DeepJet medium WP, UL2016 post-VFP
BTAG_L = 0.0480


def files_for(recid):
    m = requests.get(f"https://opendata.cern.ch/api/records/{recid}", timeout=60).json()["metadata"]
    fs = [f for idx in m.get("_file_indices", []) for f in idx["files"]]
    urls = [f["uri"].replace("root://eospublic.cern.ch/", "https://opendata.cern.ch") for f in fs]
    return urls, int(m["distribution"]["number_events"])


def dphi(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.where(d > np.pi, 2 * np.pi - d, d)


def select(a):
    """a: awkward record array of one chunk -> pandas DataFrame of selected events with features."""
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
    ht = ak.sum(jpt, axis=1)
    j1 = ak.fill_none(ak.firsts(jpt), 0.0)
    pre = (trig & flags & (nlep >= 1) & (a.MET_pt > 200) & (j1 > 100) & (ht > 200))
    a = a[pre]
    if len(a) == 0:
        return None
    mu_ok, el_ok, jet_ok = mu_ok[pre], el_ok[pre], jet_ok[pre]

    # leading (highest-pT) selected lepton, muon or electron
    def lead(pt, ok, *vals):
        idx = ak.argmax(ak.where(ok, pt, -1.0), axis=1, keepdims=True)
        return [ak.to_numpy(ak.fill_none(ak.flatten(v[idx]), np.nan)).astype(float) for v in (pt,) + vals]
    m_pt, m_eta, m_phi, m_q, m_iso, m_dxy, m_dz, m_sip = lead(
        a.Muon_pt, mu_ok, a.Muon_eta, a.Muon_phi, a.Muon_charge, a.Muon_pfRelIso03_all, a.Muon_dxy, a.Muon_dz, a.Muon_sip3d)
    e_pt, e_eta, e_phi, e_q, e_iso, e_dxy, e_dz, e_sip = lead(
        a.Electron_pt, el_ok, a.Electron_eta, a.Electron_phi, a.Electron_charge, a.Electron_pfRelIso03_all, a.Electron_dxy, a.Electron_dz, a.Electron_sip3d)
    has_m = ak.to_numpy(ak.any(mu_ok, axis=1)); has_e = ak.to_numpy(ak.any(el_ok, axis=1))
    use_m = has_m & (~has_e | (np.nan_to_num(m_pt) >= np.nan_to_num(e_pt)))
    pick = lambda x, y: np.where(use_m, x, y)
    lpt, leta, lphi = pick(m_pt, e_pt), pick(m_eta, e_eta), pick(m_phi, e_phi)
    met, metphi = ak.to_numpy(a.MET_pt).astype(float), ak.to_numpy(a.MET_phi).astype(float)

    jpt = a.Jet_pt[jet_ok]; jeta = a.Jet_eta[jet_ok]; jphi = a.Jet_phi[jet_ok]; jb = a.Jet_btagDeepFlavB[jet_ok]
    pad = lambda x, i, v=np.nan: ak.to_numpy(ak.fill_none(ak.pad_none(x, i + 1, clip=True)[:, i], v)).astype(float)
    ht = ak.to_numpy(ak.sum(jpt, axis=1)).astype(float)
    # soft secondary vertices (soft b-hadron proxy), well separated from jets not required here
    sv_ok = (a.SV_ndof >= 1.8) & (a.SV_dxySig > 3) & (a.SV_pt < 20)
    svpt = a.SV_pt[sv_ok]; svsig = a.SV_dlenSig[sv_ok]; svm = a.SV_mass[sv_ok]
    order = ak.argsort(svpt, ascending=False); svpt, svsig, svm = svpt[order], svsig[order], svm[order]

    f = pd.DataFrame({
        "lep_pt": lpt, "lep_eta": leta, "lep_is_mu": use_m.astype(float),
        "lep_charge": pick(m_q, e_q), "lep_reliso": pick(m_iso, e_iso),
        "lep_dxy": np.abs(pick(m_dxy, e_dxy)), "lep_dz": np.abs(pick(m_dz, e_dz)), "lep_sip3d": pick(m_sip, e_sip),
        "n_lep": ak.to_numpy(ak.sum(mu_ok, axis=1) + ak.sum(el_ok, axis=1)).astype(float),
        "met": met, "mt": np.sqrt(2 * lpt * met * (1 - np.cos(dphi(lphi, metphi)))),
        "dphi_lep_met": dphi(lphi, metphi), "ht": ht,
        "n_jet": ak.to_numpy(ak.num(jpt)).astype(float),
        "n_bjet_m": ak.to_numpy(ak.sum(jb > BTAG_M, axis=1)).astype(float),
        "n_bjet_l": ak.to_numpy(ak.sum(jb > BTAG_L, axis=1)).astype(float),
        "jet1_pt": pad(jpt, 0), "jet1_eta": pad(jeta, 0), "jet1_btag": pad(jb, 0),
        "jet2_pt": pad(jpt, 1), "jet2_btag": pad(jb, 1),
        "dphi_j1_met": dphi(pad(jphi, 0), metphi), "dphi_j1_j2": dphi(pad(jphi, 0), pad(jphi, 1)),
        "n_softsv": ak.to_numpy(ak.num(svpt)).astype(float),
        "sv1_pt": pad(svpt, 0), "sv1_dlensig": pad(svsig, 0), "sv1_mass": pad(svm, 0),
        "n_tau": ak.to_numpy(ak.num(a.Tau_pt)).astype(float), "n_isotrack": ak.to_numpy(a.nIsoTrack).astype(float),
        "genWeight": ak.to_numpy(a.genWeight).astype(float),
    })
    f["ct1"] = np.minimum(f.met, f.ht - 100.0)
    f["ct2"] = np.minimum(f.met, f.jet1_pt - 25.0)
    f["lep_pt_over_met"] = f.lep_pt / f.met
    return f


def process_file(job):
    """Download one file with aria2c (16 connections), select events locally, save a part."""
    name, i, url, outdir = job
    part = f"{outdir}/parts/{name}__{i}.parquet"
    if os.path.exists(part + ".json"):
        return name
    t0 = time.time()
    tmp = f"/tmp/dl/{name}__{i}.root"
    os.makedirs("/tmp/dl", exist_ok=True)
    for attempt in range(3):
        rc = os.system(f"aria2c -q -x16 -s16 -k1M --file-allocation=none -d /tmp/dl -o {name}__{i}.root '{url}'")
        if rc == 0:
            break
        if os.path.exists(tmp):
            os.remove(tmp)
    t1 = time.time()
    try:
        with uproot.open(tmp) as fh:
            runs = fh["Runs"].arrays(["genEventSumw", "genEventCount"], library="np")
            sumw, cnt = float(runs["genEventSumw"].sum()), float(runs["genEventCount"].sum())
            parts = [select(c) for c in fh["Events"].iterate(BR, step_size="200 MB")]
        parts = [p for p in parts if p is not None]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        df.to_parquet(part)
        json.dump(dict(sumw=sumw, cnt=cnt, n=len(df)), open(part + ".json", "w"))
        print(f"  {name}#{i}: sel={len(df)} dl={t1-t0:.0f}s proc={time.time()-t1:.0f}s", flush=True)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return name


def merge(name, outdir):
    proc, rec, xs = SAMPLES[name]
    js = sorted(glob.glob(f"{outdir}/parts/{name}__*.parquet.json"))
    if not js:
        return
    meta = [json.load(open(j)) for j in js]
    dfs = [pd.read_parquet(j[:-5]) for j, m in zip(js, meta) if m["n"] > 0]
    sumw, cnt = sum(m["sumw"] for m in meta), sum(m["cnt"] for m in meta)
    n_total = TOTALS[name]
    if not dfs:
        print(f"{name}: no events selected in {len(js)} files", flush=True); return
    df = pd.concat(dfs, ignore_index=True)
    df["weight"] = xs * LUMI_PB * df.genWeight / (sumw * n_total / cnt)
    df["sample"], df["process"] = name, proc
    if proc == "signal":
        ms, ml = map(float, name.split("_")[1:])
        df["m_stop"], df["m_lsp"], df["dm"] = ms, ml, ms - ml
    df["frac_read"] = cnt / n_total
    df.to_parquet(f"{outdir}/{name}.parquet")
    print(f"{name}: {len(js)} files, kept {len(df)}, frac_read={cnt/n_total:.3f}, weighted={df.weight.sum():.1f}", flush=True)


TOTALS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evt")
    ap.add_argument("--samples", default="all")
    ap.add_argument("--max_files", type=int, default=2)
    ap.add_argument("--sig_max_files", type=int, default=99)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--time_budget_h", type=float, default=10.5)
    a = ap.parse_args()
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    names = list(SAMPLES) if a.samples == "all" else a.samples.split(",")
    jobs = []
    for name in names:
        urls, TOTALS[name] = files_for(SAMPLES[name][1])
        k = a.sig_max_files if SAMPLES[name][0] == "signal" else a.max_files
        jobs += [(name, i, u, a.out) for i, u in enumerate(urls[:k])]
    print(f"{len(jobs)} files to process", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = []
        for j in jobs:
            futs.append(ex.submit(lambda jj: None if (time.time() - t0) / 3600 > a.time_budget_h else process_file(jj), j))
        for f in futs:
            try:
                f.result()
            except Exception:
                print(traceback.format_exc()[-800:], flush=True)
    for name in names:
        merge(name, a.out)


if __name__ == "__main__":
    main()
