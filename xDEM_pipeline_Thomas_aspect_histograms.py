# # -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 08:36:13 2026

@author: lguido

Post-process xDEM histogram value CSVs:
- Plot histograms for all-value and stable-terrain dDEM values
- Identify histogram peaks
- Save peak summary table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ===================================================
# SETTINGS
# ===================================================
output_dir = Path(r"\Stable5_ASPECT")

hist_dir = output_dir / "histogram_values"
plot_dir = output_dir / "histogram_peak_plots_all_and_stable"
plot_dir.mkdir(parents=True, exist_ok=True)

summary_csv = output_dir / "histogram_peak_summary_all_and_stable.csv"

# Histogram settings
BIN_WIDTH = 0.005  # meters
PLOT_RANGE = (-5, 5)  # meters


# ===================================================
# FUNCTIONS
# ===================================================
def find_histogram_peak(values, bin_width=0.005, plot_range=(-5, 5)):
    """
    Estimate histogram peak as the center of the bin with the highest count.
    """
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan, 0

    bins = np.arange(plot_range[0], plot_range[1] + bin_width, bin_width)
    counts, edges = np.histogram(values, bins=bins)

    peak_idx = np.argmax(counts)
    peak_count = counts[peak_idx]
    peak_center = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])

    return peak_center, peak_count, values.size


def make_histogram_plot(values, label, terrain_type, peak_value, peak_count, out_png):
    """
    Save histogram plot with peak marked.
    """
    values = values[np.isfinite(values)]

    bins = np.arange(PLOT_RANGE[0], PLOT_RANGE[1] + BIN_WIDTH, BIN_WIDTH)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.75)

    if np.isfinite(peak_value):
        plt.axvline(
            peak_value,
            linestyle="--",
            linewidth=2,
            label=f"Peak = {peak_value:.3f} m",
        )

    plt.axvline(0, color="k", linewidth=1, label="0 m")

    plt.xlabel("Elevation change, dh (m)")
    plt.ylabel("Pixel count")
    plt.title(
        f"{label}\n"
        f"Terrain type: {terrain_type} | "
        f"Peak = {peak_value:.3f} m, count = {peak_count:,}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ===================================================
# MAIN
# ===================================================
rows = []

csv_files = sorted(hist_dir.glob("*_dh_values.csv"))

if len(csv_files) == 0:
    raise FileNotFoundError(f"No histogram CSV files found in: {hist_dir}")

for csv_path in csv_files:
    is_stable = "_stable_dh_values" in csv_path.name
    terrain_type = "stable" if is_stable else "all"

    if is_stable:
        label = csv_path.stem.replace("_stable_dh_values", "")
    else:
        label = csv_path.stem.replace("_dh_values", "")

    df = pd.read_csv(csv_path)

    if "dh_m" not in df.columns:
        print(f"Skipping {csv_path.name}: no dh_m column")
        continue

    values = df["dh_m"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]

    peak_value, peak_count, n_values = find_histogram_peak(
        values,
        bin_width=BIN_WIDTH,
        plot_range=PLOT_RANGE,
    )

    out_png = plot_dir / f"{label}_{terrain_type}_histogram_peak.png"

    make_histogram_plot(
        values=values,
        label=label,
        terrain_type=terrain_type,
        peak_value=peak_value,
        peak_count=peak_count,
        out_png=out_png,
    )

    rows.append({
        "label": label,
        "terrain_type": terrain_type,
        "csv_file": str(csv_path),
        "plot_file": str(out_png),
        "n_values": int(n_values),
        "bin_width_m": BIN_WIDTH,
        "plot_min_m": PLOT_RANGE[0],
        "plot_max_m": PLOT_RANGE[1],
        "histogram_peak_m": peak_value,
        "histogram_peak_count": int(peak_count),
        "mean_m": np.mean(values) if values.size > 0 else np.nan,
        "median_m": np.median(values) if values.size > 0 else np.nan,
        "std_m": np.std(values) if values.size > 0 else np.nan,
        "p05_m": np.percentile(values, 5) if values.size > 0 else np.nan,
        "p25_m": np.percentile(values, 25) if values.size > 0 else np.nan,
        "p75_m": np.percentile(values, 75) if values.size > 0 else np.nan,
        "p95_m": np.percentile(values, 95) if values.size > 0 else np.nan,
    })

summary_df = pd.DataFrame(rows)

summary_df = summary_df.sort_values(
    ["label", "terrain_type"]
).reset_index(drop=True)

summary_df.to_csv(summary_csv, index=False)

print("\n========================================")
print("HISTOGRAM PEAK SUMMARY")
print("========================================")
print(summary_df)

print("\n========================================")
print("OUTPUTS SAVED")
print("========================================")
print(f"Peak summary CSV: {summary_csv}")
print(f"Histogram plots:  {plot_dir}")