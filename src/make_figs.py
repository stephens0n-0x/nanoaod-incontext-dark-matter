"""Figures for the v5 paper from v5_summary.txt numbers."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a85", "axes.labelcolor": "#3a3a38", "xtick.color": "#5a5a56", "ytick.color": "#5a5a56",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
INK = "#3a3a38"
N = np.array([10, 30, 100, 300, 1000])
T = {  # mean, se
 "Kumo Tabular eng": ([3.52,3.83,4.43,4.78,5.14],[.18,.18,.23,.27,.32]),
 "Kumo Tabular flat": ([3.16,3.67,4.29,4.71,5.07],[.14,.16,.21,.28,.32]),
 "TabICLv2 eng": ([3.25,3.55,4.01,4.36,4.51],[.16,.15,.15,.20,.23]),
 "TabICLv2 flat": ([3.05,3.37,3.90,4.26,4.38],[.13,.12,.12,.19,.24]),
 "XGBoost eng": ([2.66,3.25,3.78,4.27,4.55],[.13,.18,.18,.21,.25]),
 "XGBoost flat": ([2.39,2.99,3.51,3.96,4.18],[.10,.17,.15,.16,.18]),
 "Kumo Relational + eng": ([2.94,3.30,3.62,4.00,4.19],[.14,.13,.12,.15,.16]),
 "Kumo Relational": ([2.71,3.24,3.53,3.86,4.13],[.12,.13,.13,.14,.17]),
 "DeepSets": ([1.74,2.41,3.10,3.28,3.11],[.12,.15,.14,.15,.10])}
STY = {"Kumo Tabular eng": (BLUE,"-","o"), "Kumo Tabular flat": (BLUE,"--","o"), "TabICLv2 eng": (AQUA,"-","s"),
       "TabICLv2 flat": (AQUA,"--","s"), "XGBoost eng": (ORANGE,"-","^"), "XGBoost flat": (ORANGE,"--","^"),
       "Kumo Relational + eng": (VIOLET,"-","D"), "Kumo Relational": (VIOLET,"--","D"), "DeepSets": (MAGENTA,"--","v")}

def save(fig, name):
    fig.savefig(f"../figures/{name}.pdf"); fig.savefig(f"../figures/{name}.png", dpi=220); plt.close(fig)

# ---------------- Fig 1: scaling
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, constrained_layout=True)
for ax, keys, title in ((axs[0], ["Kumo Tabular eng","Kumo Tabular flat","TabICLv2 eng","TabICLv2 flat","XGBoost eng","XGBoost flat"], "(a) Tabular inputs: in-context vs. trained"),
                        (axs[1], ["Kumo Tabular flat","Kumo Relational + eng","Kumo Relational","DeepSets"], "(b) Object-level inputs")):
    for k in keys:
        c, ls, mk = STY[k]; m, s = map(np.array, T[k])
        ax.fill_between(N, m - s, m + s, color=c, alpha=0.12, lw=0)
        ax.plot(N, m, ls=ls, color=c, lw=2, marker=mk, ms=4.5, mfc="white" if ls == "--" else c, mew=1.3, label=k)
    ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels(N); ax.minorticks_off()
    ax.set_xlabel("signal events in context, N"); ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7.2, loc="lower right", handlelength=2.6)
axs[0].set_ylabel(r"$G_{\rm val}$ (mean $\pm$ s.e.)")
save(fig, "fig1_scaling")

# ---------------- Fig 2: forest plot of paired ratios
rows = [("Prior, same features", [("Kumo Tabular eng / XGBoost eng",1.177,1.13,1.23),("TabICLv2 eng / XGBoost eng",1.078,1.04,1.12)]),
        ("Prior, same raw objects", [("Kumo Tabular flat / XGBoost flat",1.225,1.17,1.28),("TabICLv2 flat / XGBoost flat",1.127,1.08,1.17)]),
        ("Engineering needed?", [("Kumo Tabular flat / XGBoost eng",1.130,1.10,1.17),("TabICLv2 flat / XGBoost eng",1.039,1.00,1.08),
                                 ("Kumo Tabular flat / Kumo Tabular eng",0.960,0.94,0.98),("TabICLv2 flat / TabICLv2 eng",0.964,0.95,0.98)]),
        ("Which prior", [("Kumo Tabular / TabICLv2, flat",1.087,1.06,1.12),("Kumo Tabular / TabICLv2, eng",1.091,1.06,1.12)]),
        ("Structure", [("Kumo Tabular flat / Kumo Relational",1.177,1.13,1.23),("Kumo Relational / XGBoost flat",1.041,1.01,1.07),
                       ("Kumo Relational / XGBoost eng",0.959,0.93,0.99),("Kumo Relational / DeepSets",1.315,1.24,1.41)]),
        ("Engineering inside relational", [("Kumo Rel.+eng / Kumo Relational",1.036,1.02,1.05),("Kumo Rel.+eng / Kumo Tabular eng",0.844,0.80,0.88)])]
GC = [BLUE, ORANGE, AQUA, YELLOW, VIOLET, MAGENTA]
n = sum(len(r) for _, r in rows) + len(rows)
fig, ax = plt.subplots(figsize=(6.4, 0.26 * n + 0.6), constrained_layout=True)
y = 0; yt, yl = [], []
for gi, (g, items) in enumerate(rows):
    ax.text(0.775, -y, g, fontsize=8.2, fontweight="bold", color=INK, va="center"); y += 1
    for lab, r, lo, hi in items:
        ax.plot([lo, hi], [-y, -y], color=GC[gi], lw=2, solid_capstyle="round")
        ax.plot(r, -y, "o", color=GC[gi], ms=5.5, mec="white", mew=1.2)
        yt.append(-y); yl.append(lab); y += 1
ax.axvline(1.0, color="#8a8a85", ls="--", lw=1)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=7.8); ax.set_xlim(0.77, 1.45); ax.grid(axis="y", visible=False)
ax.set_xlabel("paired ratio of $G_{\\rm val}$ (geometric mean, 95% CI)")
save(fig, "fig2_forest")

# ---------------- Fig 3: ablation + LOO
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.6]}, constrained_layout=True)
ab = [("no leptons table",0.555,0.53,0.58),("no jets table",0.967,0.94,0.99),("no secondary vertices",1.010,0.99,1.03),
      ("no isolated tracks",1.033,1.02,1.05),("2 hops, star schema",1.017,1.00,1.04)]
ax = axs[0]
for i, (lab, r, lo, hi) in enumerate(ab):
    ax.plot([lo, hi], [-i, -i], color=VIOLET, lw=2, solid_capstyle="round"); ax.plot(r, -i, "o", color=VIOLET, ms=5.5, mec="white", mew=1.2)
    ax.text(hi + 0.02, -i, f"{r:.2f}", va="center", fontsize=7.5, color=INK)
ax.axvline(1, color="#8a8a85", ls="--", lw=1); ax.set_yticks(-np.arange(len(ab))); ax.set_yticklabels([a[0] for a in ab], fontsize=7.8)
ax.set_xlim(0.45, 1.15); ax.grid(axis="y", visible=False); ax.set_xlabel("ratio to full Kumo Relational")
ax.set_title("(a) Removing one object table", fontsize=9, loc="left", color=INK)
pts = ["(300,290)","(350,335)","(400,380)","(500,490)","(500,420)","(525,438)","(525,350)"]
L = {"Kumo Tabular eng":[3.17,3.47,2.77,3.19,1.72,2.23,1.43], "Kumo Tabular flat":[3.40,3.58,2.74,3.17,1.60,2.28,1.64],
     "TabICLv2 eng":[3.22,3.29,2.68,2.94,2.39,2.37,1.60], "TabICLv2 flat":[3.04,3.25,2.45,2.77,2.93,2.68,1.51],
     "XGBoost eng":[2.95,3.18,2.51,2.34,1.58,2.39,1.52], "PNN":[2.85,2.83,2.37,2.20,1.98,1.77,1.54], "Kumo Relational":[2.44,2.91,1.93,2.27,1.50,1.32,1.11]}
LS = {"Kumo Tabular eng":(BLUE,"o",True),"Kumo Tabular flat":(BLUE,"o",False),"TabICLv2 eng":(AQUA,"s",True),"TabICLv2 flat":(AQUA,"s",False),
      "XGBoost eng":(ORANGE,"^",True),"PNN":(YELLOW,"P",True),"Kumo Relational":(VIOLET,"D",False)}
ax = axs[1]; x = np.arange(7); off = np.linspace(-0.27, 0.27, len(L))
for j, (k, v) in enumerate(L.items()):
    c, mk, filled = LS[k]
    ax.plot(x + off[j], v, mk, color=c, ms=5, mfc=c if filled else "white", mew=1.3, ls="none", label=k)
for b in (2.5, 4.5): ax.axvline(b, color="#cfcec8", lw=0.8)
for xs, name in ((1, "LLStop"), (3.5, "T2-4bd"), (5.5, "T2tt")): ax.text(xs, 3.85, name, ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(pts, fontsize=7.2, rotation=25); ax.set_ylim(1.0, 4.0); ax.grid(axis="x", visible=False)
ax.set_ylabel(r"$G_{\rm val}$, unseen point (5-seed mean)"); ax.set_title("(b) Transfer to an unseen mass point", fontsize=9, loc="left", color=INK)
h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncol=7, fontsize=6.8, handletextpad=0.2, columnspacing=0.9)
save(fig, "fig3_ablation_transfer")
print("ok")
