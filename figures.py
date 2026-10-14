import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results.json"))
CH = [("reading", "Reading\n(9 regions)"),
      ("maths_ops", "Mathematics: operations\nladder (12 regions)"),
      ("maths_num", "Mathematics: number\nrecognition (12 regions)")]
INK, MUTED, GRID, MARK = "#1f2328", "#5b636b", "#e6e8eb", "#2f6db5"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

def title(s):
    return s.title().replace("'I", "'i").replace("Vakinakaratra", "Vakinankaratra")

# Figure 3: forest plot of rho with 95% cluster-bootstrap intervals
fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharex=True)
for ax, (k, lab) in zip(axes, CH):
    res = R[k]["res"]; regs = sorted([r for r in res if not r.startswith("_") and r != "NATIONAL"],
                                     key=lambda r: res[r]["rho"])
    y = np.arange(len(regs))
    rho = np.array([res[r]["rho"] for r in regs]); ci = np.array([res[r]["rho_ci"] for r in regs])
    ax.hlines(y, ci[:, 0], ci[:, 1], color=MARK, lw=2)
    ax.plot(rho, y, "o", ms=6, color=MARK, mec="white", mew=1.5, zorder=3)
    nat = res["NATIONAL"]
    ax.axvspan(*nat["rho_ci"], color=GRID, zorder=0)
    ax.axvline(nat["rho"], color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_yticks(y, [title(r) for r in regs])
    ax.set_title(lab, fontsize=9, color=INK, loc="left")
    ax.set_xlim(0, 1); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.set_xlabel(r"Spectral radius $\rho(\hat Q)$")
fig.tight_layout()
fig.savefig("fig_rho_regions.pdf"); fig.savefig("fig_rho_regions.png", dpi=200)

# Figure 4: rho vs expected number of intervention periods to mastery from the lowest level
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, (k, lab) in zip(axes, CH):
    res = R[k]["res"]; regs = [r for r in res if not r.startswith("_") and r != "NATIONAL"]
    x = np.array([res[r]["rho"] for r in regs]); y = np.array([res[r]["t"][0] for r in regs])
    ax.plot(x, y, "o", ms=6, color=MARK, mec="white", mew=1.5)
    for r, xi, yi in zip(regs, x, y):
        if yi == y.max() or yi == y.min():
            ax.annotate(title(r), (xi, yi), xytext=(4, 3), textcoords="offset points", fontsize=7, color=MUTED)
    mo = res["_mono"]
    ax.set_title(f"{lab.split(' (')[0].replace(chr(10),' ')}\nSpearman = {mo['spearman']:.2f}", fontsize=9, color=INK, loc="left")
    ax.set_xlabel(r"$\rho(\hat Q)$"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
axes[0].set_ylabel(r"$\mathbb{E}_{1}[T_M]$ (intervention periods)")
fig.tight_layout()
fig.savefig("fig_rho_vs_time.pdf"); fig.savefig("fig_rho_vs_time.png", dpi=200)
print("ok")
