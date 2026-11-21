# Representation vs. prior: in-context foundation models for few-shot dark-matter searches on CMS NanoAOD

Code for the study *"Representation vs. prior: do in-context foundation models remove feature engineering in few-shot dark-matter searches? A study on native CMS NanoAOD"*.

We compare pretrained in-context foundation models used **without any training** (NVIDIA Kumo Tabular, NVIDIA Kumo Relational, TabICLv2) with trained models (XGBoost, DeepSets, a Set Transformer, a parameterised neural network) for compressed top-squark → neutralino (dark-matter) signals, on three representations of the same CMS Open Data events:

- **engineered**: 32 hand-built physics variables;
- **flat**: missing p_T plus the leading reconstructed objects as padded columns, with no derived variables;
- **relational**: the native NanoAOD structure, an events table linked to jet, lepton, secondary-vertex and track tables, optionally with jet–object links (ΔR < 0.4).

## Main results (v5: 4 mass points × 5 signal sizes × 5 seeds, single top included)

| Comparison | Wins | Sensitivity ratio [95% CI] |
|---|---|---|
| Kumo Tabular **flat** vs XGBoost **engineered** | 90/100 | 1.13 [1.10, 1.17] |
| Kumo Tabular flat vs Kumo Tabular engineered | 29/100 | 0.96 [0.94, 0.98] |
| TabICLv2 flat vs XGBoost engineered | 63/100 | 1.04 [1.00, 1.08] |
| Kumo Tabular flat vs Kumo Relational | 94/100 | 1.18 [1.13, 1.23] |
| Kumo Relational vs DeepSets | 94/100 | 1.32 [1.24, 1.41] |
| Kumo Relational without lepton table vs full | 0/60 | 0.56 [0.53, 0.58] |

Full numbers are in [`results/v5_summary.txt`](results/v5_summary.txt) and the figures in [`figures/`](figures/).

## Data

All inputs are public CMS Open Data (CERN Open Data portal), RunIISummer20UL16 NanoAODv9 simulation:

- signal records 63443, 63445, 63451, 63453, 63455, 63460, 63462;
- backgrounds: W+jets, tt̄, Z→νν+jets, WZ→3ℓν (records 67801, 67993, 69723–69735, 72750, 74896–74906) and single top (64659, 64759, 64825, 64881).

Files are streamed from the portal; nothing is redistributed here.

## Code

| File | Purpose |
|---|---|
| `src/prep.py` | sample list, cross sections, event selection, 32 engineered features |
| `src/prep_rel.py` | relational extraction: event table + jets / leptons / secondary vertices / tracks, jet links |
| `src/exp.py` | shared utilities: loading, importance-thinned test background, metrics, Kumo Tabular and XGBoost wrappers |
| `src/exp_rel.py` | flat (top-k) representation and the Kumo Relational schema |
| `src/exp_rel2.py` | the full study: all models, few-shot / table-removal / unseen-mass-point modes, validation-chosen working point |
| `src/run_v5.py` | driver used for the final run on Kaggle (2× T4) |
| `src/an5.py` | paired comparisons with cluster-bootstrap 95% intervals |
| `src/make_figs.py` | the three paper figures |

## Running

```bash
pip install -r requirements.txt
cd src
# 1. extract relational parquet tables (CPU, streams files from opendata.cern.ch)
python prep_rel.py --out rel --samples <comma-separated sample names from prep.SAMPLES> --max_files 2
# 2. run the study on a GPU (example: few-shot, two mass points)
python exp_rel2.py --indir rel --mode fewshot --points sig_300_290,sig_350_335 \
    --nsig 10,30,100,300,1000 --seeds 5 --models tab_flat,tab_eng,ticl_flat,xgb_eng,rel --out results.csv
# 3. paired comparisons and intervals
python an5.py results.csv
```

`run_v5.py` reproduces the exact v5 configuration on Kaggle; it expects the extracted tables under `/kaggle/input`.

## Metric

Sensitivity gain G = max ε_s/√ε_b over score thresholds with at least 20 raw background events above threshold. The threshold is chosen on one half of the test set and evaluated on the other half (G_val). G does not depend on the signal cross section or generator-filter efficiency.

## Licence

MIT, see [`LICENSE`](LICENSE). The pretrained models keep their own licences (NVIDIA open-model licence, TabICL licence).
