from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
s = json.loads((root / "final/summary.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
colors = ["#d88738", "#939ba4", "#16718b"]
labels = ["Old: selected-family BH", "Old: full-family BH", "New: full-family BH/e-BH"]
for ax, signal, keys, title in zip(
    axes,
    [0, 3],
    [
        ["diagnostic_bh_reject", "diagnostic_full_bh_false_any", "bh_false_any"],
        ["diagnostic_true_any", "diagnostic_full_bh_true_any", "bh_true_any"],
    ],
    ["Null: any false discovery", "Injected sites: at least one detected"],
):
    rows = [
        r for r in s["conditions"] if not r["filtered"] and r["signal_sites"] == signal
    ]
    x = np.arange(3)
    for j, (key, color, label) in enumerate(zip(keys, colors, labels)):
        values = np.array([r["metrics"][key]["mean"] * 100 for r in rows])
        ci = np.array([r["metrics"][key]["ci95"] for r in rows]).T * 100
        ax.bar(x + (j - 1) * 0.23, values, 0.21, color=color, label=label, zorder=3)
        ax.errorbar(
            x + (j - 1) * 0.23,
            values,
            yerr=np.maximum(0, np.vstack((values - ci[0], ci[1] - values))),
            fmt="none",
            ecolor=color,
            capsize=3,
            zorder=4,
        )
    ax.set_xticks(x, ["Sparse", "Unequal branches", "Uncertain ASR"])
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_ylabel("Datasets (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_ylim(0, 15 if signal == 0 else 43)
    if signal == 0:
        ax.axhline(5, color="#555", linestyle="--", linewidth=1, label="Nominal 5%")
axes[1].legend(loc="upper left", fontsize=8, frameon=False)
fig.suptitle(
    "Known CTMC parameters · 1,000 datasets per condition · 95% exact intervals",
    fontsize=12,
)
fig.savefig(root / "calibration.png", dpi=160)
fig.savefig(root / "calibration.pdf")
