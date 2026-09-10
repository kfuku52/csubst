#!/usr/bin/env python3
"""Render the saved scan endpoint accuracy and timing comparisons."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.outdir / "accuracy.tsv", sep="\t")
    colors = {"legacy_raw": "#2277aa", "legacy_n": "#cc7733"}
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.6), sharex=True, sharey=True)
    for row, candidate in enumerate(["F->K", "F->L"]):
        for col, model in enumerate(["GY", "ECMrest", "ECMK07"]):
            ax = axes[row, col]
            frame = data[(data.model == model) & (data.candidate == candidate)]
            ax.axhline(1, color="#222222", linewidth=1.8, label="Endpoint (exact reference)")
            for field, label, style in [("legacy_raw", "Q-weighted / raw", "-"),
                                        ("legacy_n", "Q-weighted / N-rescaled", "--")]:
                ax.plot(frame.t, frame[field]/frame.endpoint_exact, style,
                        color=colors[field], marker="o", markersize=4, label=label)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=.1)
            ax.set_title(model + "  |  " + candidate, fontsize=11)
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 1:
                ax.set_xlabel("Model branch length")
    axes[0, 0].set_yticks([0, .1, 1, 10, 100], ["0", "0.1", "1", "10", "100"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Finite-time exposure consistency: 1 is the endpoint target", fontsize=15)
    fig.supylabel("Exposure / exact endpoint probability", fontsize=12)
    fig.tight_layout(rect=(.015, .065, 1, .95))
    fig.savefig(args.outdir / "accuracy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    modes = ["legacy_default", "q_raw", "endpoint"]
    for ax, filename, title in zip(axes, ["performance_none.json", "performance_full.json"],
                                   ["Scan without permutations", "Scan + 20 full_scan permutations"]):
        summary = json.loads((args.outdir / filename).read_text())["summary"]
        y = np.array([summary[m]["process_seconds"]["median"] for m in modes])
        low = np.array([summary[m]["process_seconds"]["minimum"] for m in modes])
        high = np.array([summary[m]["process_seconds"]["maximum"] for m in modes])
        ax.bar(range(3), y, color=["#cc7733", "#2277aa", "#444444"],
               yerr=np.vstack((y-low, high-y)), capsize=4)
        ax.set_xticks(range(3), ["Legacy default\nN-rescaled", "Q-weighted\nraw", "Endpoint\nraw"])
        ax.set_ylabel("Wall time (seconds)")
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(high)*1.23)
        for i, value in enumerate(y):
            ax.text(i, high[i] + max(high)*.025, f"{value:.2f}s", ha="center", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PEPC, 98 unchanged candidates, one BLAS thread", fontsize=14)
    fig.text(.5, .015, "Median of 3 runs; error bars show min-max. One warmup per mode excluded. ASR fit excluded.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .93))
    fig.savefig(args.outdir / "runtime.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
