# DemDifferencing

## Overview

This repository contains scripts, outputs, and summary products used to evaluate and improve DEM coregistration and bias correction for the Montecito study area.

The workflow uses xDEM to:

* Coregister DEMs using Nuth & Kääb + ICP
* Apply polynomial deramping
* Apply directional bias correction
* Apply terrain-based aspect bias correction
* Evaluate uncertainty using the Hugonnet et al. (2022) framework
* Compare correction strategies through sensitivity analyses
* Export dDEM values for histogram-based benchmarking

---

## Repository Contents

### Scripts

**Sensitivity Analysis (Aspect Bias Included)**

* Tests multiple deramp orders and directional bias angles
* Includes aspect bias correction
* Generates performance metrics used to identify the preferred correction workflow

**Production Processing Script**

* Applies the preferred correction workflow identified through sensitivity testing:

  * Nuth & Kääb + ICP coregistration
  * First-order deramp
  * Directional bias correction (90°)
  * Aspect bias correction

**Histogram Analysis Script**

* Reads exported dDEM values
* Generates histogram plots
* Identifies histogram peak locations
* Produces summary statistics for comparison with collaborator workflows

---

## Included Outputs

### Histogram Products

`histogram_peak_plots_all_and_stable/`

Contains histogram plots for:

* All valid pixels within the AOI
* Stable-terrain pixels only

### Summary Tables

`histogram_peak_summary_all_and_stable.csv`

* Histogram peak location
* Mean
* Median
* Standard deviation
* Selected percentiles
* Reported separately for all pixels and stable terrain

`xdem_run_metrics_raw.csv`

* Per-run performance metrics for all sensitivity-analysis runs

`xdem_run_metrics_ranked.csv`

* Ranked performance table used to identify preferred correction parameters

`xdem_uncertainty_summary.csv`

* Uncertainty estimates and associated summary statistics

---

## Notes

* File paths and local machine information have been removed from outputs prior to sharing.
* Stable terrain was defined using a threshold on the initial raw dDEM.
* Histogram outputs are provided both for benchmarking against collaborator workflows and for evaluating correction performance.

---

## Current Preferred Workflow

Based on the sensitivity analyses included here:

* Coregistration: Nuth & Kääb + ICP
* Deramp: First-order polynomial
* Directional bias correction: 90°
* Aspect bias correction: Enabled

Additional testing and refinement are welcome.
