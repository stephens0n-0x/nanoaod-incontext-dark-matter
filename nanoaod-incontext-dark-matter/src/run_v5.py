"""v5 driver: single top extraction + TabICLv2 + 5 seeds everywhere; full results written as .csv AND .txt."""
import glob, os, shutil, subprocess, sys, time

W = "/kaggle/working"
os.chdir(W)


def sh(c):
    print(">>", c, flush=True)
    return subprocess.call(c, shell=True)


sh(f"{sys.executable} -m pip install -q structured-data-models tabicl uproot awkward xgboost 2>&1 | tail -2")

# ---- 1. data: v4 relational parquet (with jet links) + new single-top samples
src = os.path.dirname(glob.glob("/kaggle/input/**/sig_300_290__svs.parquet", recursive=True)[0])
os.makedirs("rel5", exist_ok=True)
for f in glob.glob(f"{src}/*.parquet"):
    d = f"rel5/{os.path.basename(f)}"
    if not os.path.exists(d):
        os.symlink(f, d)
ST = "ST_tW_top,ST_tW_antitop,ST_t_top,ST_t_antitop"
if not os.path.exists("rel5/ST_t_antitop.parquet"):
    sh(f"{sys.executable} prep_rel.py --out st --samples {ST} --max_files 3 --workers 4 --time_budget_h 1.0 2>&1 | tail -30")
    for f in glob.glob("st/*.parquet"):
        shutil.copy(f, "rel5/")
import pandas as pd
for f in sorted(glob.glob("rel5/ST_*.parquet")):
    if "__" not in f:
        d = pd.read_parquet(f); print(f, len(d), "events, weighted", round(d.weight.clip(lower=0).sum(), 1), flush=True)

# ---- 2. smoke test of every model on one tiny cell
sh(f"CUDA_VISIBLE_DEVICES=0 {sys.executable} exp_rel2.py --indir rel5 --out smoke.csv --points sig_400_380 --nsig 30 "
   f"--seeds 1 --seed0 99 --models ticl_flat,ticl_eng,tab_flat,rel,pnn 2>&1 | tail -8")

# ---- 3. two GPU workers, seed loop outermost so a time-out loses whole seeds evenly
FS = "--mode fewshot --nsig 10,30,100,300,1000 --seeds 5"
FM = "--models tab_flat,tab_eng,ticl_flat,ticl_eng,xgb_flat,xgb_eng,rel,rel_eng,deepsets"
AB = "--mode fewshot --nsig 30,100,300 --seeds 5 --models rel,rel_noLep,rel_noSV,rel_noTrk,rel_noJet,rel_h2"
LO = "--mode loo --seeds 5 --models tab_eng,tab_flat,ticl_eng,ticl_flat,xgb_eng,pnn,rel,rel_eng"
P = "python exp_rel2.py --indir rel5"
J = {
    0: f"{P} {FS} {FM} --points sig_300_290,sig_350_335 --out f0.csv --time_budget_h 5.2 > log_f0.txt 2>&1; "
       f"{P} {AB} --points sig_300_290,sig_350_335,sig_400_380,sig_500_490 --out abl.csv --time_budget_h 2.0 > log_abl.txt 2>&1; "
       f"{P} {LO} --points sig_300_290,sig_350_335,sig_400_380 --out loo0.csv --time_budget_h 2.0 > log_loo0.txt 2>&1",
    1: f"{P} {FS} {FM} --points sig_400_380,sig_500_490 --out f1.csv --time_budget_h 5.2 > log_f1.txt 2>&1; "
       f"{P} {LO} --points sig_500_490,sig_500_420,sig_525_438,sig_525_350 --out loo1.csv --time_budget_h 3.8 > log_loo1.txt 2>&1",
}
procs = {g: subprocess.Popen(f'CUDA_VISIBLE_DEVICES={g} sh -c "{c}"', shell=True) for g, c in J.items()}
t0 = time.time()
while any(p.poll() is None for p in procs.values()):
    time.sleep(300)
    n = {f: sum(1 for _ in open(f)) - 1 for f in glob.glob("*.csv") if f != "smoke.csv"}
    print(f"{(time.time() - t0) / 3600:.2f} h", n, flush=True)

# ---- 4. full results as text (the output viewer truncates csv previews at 50 rows)
import pandas as pd
fs = [f for f in ("f0.csv", "f1.csv", "abl.csv", "loo0.csv", "loo1.csv") if os.path.exists(f)]
allr = pd.concat([pd.read_csv(f).assign(src=f) for f in fs], ignore_index=True)
allr.to_csv("results_v5.csv", index=False)
allr.to_csv("results_v5.txt", index=False)
print("done", len(allr), "rows;", allr.error.notna().sum(), "errors", flush=True)
