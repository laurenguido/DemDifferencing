# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:06:29 2026

@author: lguido
"""

import xdem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import geoutils as gu


# ===================================================
# SETTINGS
# ===================================================
DEM_PATHS = {
    "pre": Path(r"pre_step_010_class_2.tif"),
    "post": Path(r"post_step_010_class_2.tif"),
}

AOI_PATH = Path(r"montecito_aoi.shp")

REFERENCE_LABEL = "pre"

# Production correction parameters from sensitivity analysis
DERAMP_ORDER = 1
DIRECTIONAL_ANGLE = 90

STABLE_TERRAIN_DH_THRESHOLD = 5.0  # meters
MAP_CLIM = 1.0  # +/- meters

ASPECT_BIN_EDGES = np.linspace(0, 360, 73)  # 5-degree aspect bins

output_dir = Path(r"Production_o1_a90_ASPECT")


# ===================================================
# UTILITY FUNCTIONS
# ===================================================
def safe_array(dem_or_dh):
    arr = dem_or_dh.data.copy()

    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    if hasattr(dem_or_dh, "nodata") and dem_or_dh.nodata is not None:
        arr[arr == dem_or_dh.nodata] = np.nan

    return arr.astype(float)


def robust_nmad(arr):
    med = np.median(arr)
    return 1.4826 * np.median(np.abs(arr - med))


def get_aspect(dem):
    try:
        return dem.get_terrain_attribute("aspect")
    except AttributeError:
        return xdem.terrain.aspect(dem, resolution=dem.res[0])


# ===================================================
# PIPELINE OBJECT
# ===================================================
class CoregBiasPipeline:

    def __init__(
        self,
        reference_label,
        deramp_order=1,
        directional_angle=90,
        stable_dh_threshold=5.0,
        map_clim=1.0,
    ):
        self.reference_label = reference_label
        self.deramp_order = deramp_order
        self.directional_angle = directional_angle
        self.stable_dh_threshold = stable_dh_threshold
        self.map_clim = map_clim

        self.metadata = {}
        self.results = []
        self.stable_masks = {}

    # ---------------------------------------------------
    # DEM loading / alignment / clipping
    # ---------------------------------------------------
    def load_dems(self, dem_paths, aoi_path=None):
        self.dems = {k: xdem.DEM(v) for k, v in dem_paths.items()}
        self.ref = self.dems[self.reference_label]

        square_res = float(self.ref.res[0])
        self.ref = self.ref.reproject(res=square_res)
        self.dems[self.reference_label] = self.ref

        self.dems = {
            k: (v if k == self.reference_label else v.reproject(self.ref))
            for k, v in self.dems.items()
        }

        bounds = [d.bounds for d in self.dems.values()]
        common_bounds = (
            max(b.left for b in bounds),
            max(b.bottom for b in bounds),
            min(b.right for b in bounds),
            min(b.top for b in bounds),
        )

        self.dems = {k: v.crop(common_bounds) for k, v in self.dems.items()}
        self.ref = self.dems[self.reference_label]

        if aoi_path is not None:
            aoi = gu.Vector(aoi_path)
            aoi = aoi.to_crs(self.ref.crs)

            aoi_mask = aoi.create_mask(self.ref)
            aoi_mask_arr = aoi_mask.data

            if np.ma.isMaskedArray(aoi_mask_arr):
                aoi_mask_arr = aoi_mask_arr.filled(False)

            aoi_mask_arr = np.asarray(aoi_mask_arr).astype(bool)

            for key, dem in self.dems.items():
                arr = dem.data.copy()

                if np.ma.isMaskedArray(arr):
                    arr = arr.filled(np.nan)

                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr[0]

                arr[~aoi_mask_arr] = np.nan
                dem.data = np.ma.masked_invalid(arr)

                self.dems[key] = dem

            self.ref = self.dems[self.reference_label]
            print(f"Applied AOI mask: {aoi_path}")

        self.metadata["load"] = {
            "DEM keys": list(self.dems.keys()),
            "reference_label": self.reference_label,
            "stable_dh_threshold_m": self.stable_dh_threshold,
            "aoi_path": str(aoi_path) if aoi_path is not None else None,
        }

        print(f"Loaded, aligned, cropped, and clipped DEMs: {list(self.dems.keys())}")

    # ---------------------------------------------------
    # Stable terrain mask
    # ---------------------------------------------------
    def build_stable_mask(self, pair_name, tgt):
        print(f"\nBuilding stable terrain mask for {pair_name}")

        dh_raw = tgt - self.ref
        dh_raw_arr = safe_array(dh_raw)

        slope = xdem.terrain.slope(self.ref, resolution=self.ref.res[0])
        slope_arr = safe_array(slope)

        valid_mask = np.isfinite(dh_raw_arr) & np.isfinite(slope_arr)
        stable_mask = valid_mask & (np.abs(dh_raw_arr) < self.stable_dh_threshold)

        self.stable_masks[pair_name] = stable_mask

        stable_count = np.count_nonzero(stable_mask)
        valid_count = np.count_nonzero(valid_mask)
        stable_pct = 100 * stable_count / valid_count if valid_count > 0 else np.nan

        print(f"Stable threshold: |dh_raw| < {self.stable_dh_threshold:.3f} m")
        print(f"Stable pixels: {stable_count:,} / {valid_count:,}")
        print(f"Stable fraction: {stable_pct:.2f}%")

        self.metadata[f"{pair_name}_stable_mask"] = {
            "threshold_m": self.stable_dh_threshold,
            "stable_pixels": int(stable_count),
            "valid_pixels": int(valid_count),
            "stable_fraction_percent": float(stable_pct),
        }

        plt.figure(figsize=(7, 6))
        plt.imshow(dh_raw_arr, cmap="RdBu", vmin=-self.map_clim, vmax=self.map_clim)
        plt.colorbar(label="Initial elevation change, dh_raw = target - reference (m)")
        plt.title(f"{pair_name} - Initial Raw dDEM")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 6))
        plt.imshow(stable_mask, cmap="gray")
        plt.colorbar(label="Stable terrain mask")
        plt.title(
            f"{pair_name} - Stable Terrain Mask\n"
            f"|dh_raw| < {self.stable_dh_threshold:.2f} m, "
            f"stable area = {stable_pct:.2f}%"
        )
        plt.tight_layout()
        plt.show()

        return stable_mask

    # ---------------------------------------------------
    # Coregistration
    # ---------------------------------------------------
    def coregister(self, tgt, stable_mask):
        print("\nRunning coregistration: NuthKaab + ICP")

        coreg = xdem.coreg.NuthKaab() + xdem.coreg.ICP()
        coreg.fit(self.ref, tgt, inlier_mask=stable_mask)
        tgt_coreg = coreg.apply(tgt)

        self.metadata["coreg"] = {
            "method": "NuthKaab + ICP",
            "used_stable_mask": True,
            "stable_threshold_m": self.stable_dh_threshold,
        }

        return tgt_coreg

    # ---------------------------------------------------
    # Production bias correction
    # ---------------------------------------------------
    def bias_correct_production(self, tgt, stable_mask):
        print(
            f"\nRunning production bias correction: "
            f"deramp_order={self.deramp_order}, "
            f"directional_angle={self.directional_angle}, "
            f"aspect=True"
        )

        try:
            deramp = xdem.coreg.Deramp(poly_order=self.deramp_order)

            dirbias = xdem.coreg.DirectionalBias(
                angle=self.directional_angle,
                fit_or_bin="bin",
                bin_sizes=1000,
                bin_apply_method="per_bin",
                bin_statistic=np.nanmedian,
            )

            aspect_bias = xdem.coreg.TerrainBias(
                terrain_attribute="aspect",
                fit_or_bin="bin",
                bin_sizes={"aspect": ASPECT_BIN_EDGES},
                bin_apply_method="per_bin",
                bin_statistic=np.nanmedian,
            )

            aspect = get_aspect(self.ref)

            deramp.fit(self.ref, tgt, inlier_mask=stable_mask)
            tgt_deramped = deramp.apply(tgt)

            dirbias.fit(self.ref, tgt_deramped, inlier_mask=stable_mask)
            tgt_dirbias = dirbias.apply(tgt_deramped)

            aspect_bias.fit(
                self.ref,
                tgt_dirbias,
                inlier_mask=stable_mask,
                bias_vars={"aspect": aspect},
            )

            tgt_corrected = aspect_bias.apply(
                tgt_dirbias,
                bias_vars={"aspect": aspect},
            )

            self.metadata["production_bias"] = {
                "status": "success",
                "deramp_order": self.deramp_order,
                "directional_angle": self.directional_angle,
                "aspect_bias": True,
                "aspect_bin_degrees": 5,
            }

            return tgt_corrected

        except Exception as e:
            print("\nProduction bias correction failed")
            print(f"Reason: {e}")

            self.metadata["production_bias"] = {
                "status": "failed",
                "deramp_order": self.deramp_order,
                "directional_angle": self.directional_angle,
                "aspect_bias": True,
                "error": str(e),
            }

            return None

    # ---------------------------------------------------
    # dDEM analysis / plotting / histogram exports
    # ---------------------------------------------------
    def analyze(self, label, tgt, stage="unknown", stable_mask=None):
        dh = tgt - self.ref
        dh_arr = safe_array(dh)

        valid = np.isfinite(dh_arr)
        arr = dh_arr[valid]

        if arr.size == 0:
            print(f"No valid data found for {label}")
            return

        hist_dir = output_dir / "histogram_values"
        hist_dir.mkdir(parents=True, exist_ok=True)

        safe_label = label.replace("/", "_").replace("\\", "_").replace(":", "_")

        hist_csv = hist_dir / f"{safe_label}_dh_values.csv"

        pd.DataFrame({
            "dh_m": arr
        }).to_csv(hist_csv, index=False)

        stable_csv = None
        stable_vals = np.array([])

        if stable_mask is not None:
            stable_vals = dh_arr[stable_mask]
            stable_vals = stable_vals[np.isfinite(stable_vals)]

            stable_csv = hist_dir / f"{safe_label}_stable_dh_values.csv"

            pd.DataFrame({
                "dh_m": stable_vals
            }).to_csv(stable_csv, index=False)

        stats = {
            "pair": label,
            "stage": stage,
            "deramp_order": self.deramp_order if stage == "coreg_bias_aspect" else None,
            "directional_angle": self.directional_angle if stage == "coreg_bias_aspect" else None,
            "aspect_bias": stage == "coreg_bias_aspect",
            "mean": np.mean(arr),
            "median": np.median(arr),
            "std": np.std(arr),
            "NMAD": robust_nmad(arr),
            "RMSE": np.sqrt(np.mean(arr**2)),
            "n_valid": int(arr.size),
            "dh_values_csv": str(hist_csv),
            "stable_dh_values_csv": str(stable_csv) if stable_csv is not None else None,
        }

        if stable_mask is not None:
            if stable_vals.size > 0:
                stats["stable_mean"] = np.mean(stable_vals)
                stats["stable_median"] = np.median(stable_vals)
                stats["stable_std"] = np.std(stable_vals)
                stats["stable_NMAD"] = robust_nmad(stable_vals)
                stats["stable_RMSE"] = np.sqrt(np.mean(stable_vals**2))
                stats["n_stable"] = int(stable_vals.size)
            else:
                stats["stable_mean"] = np.nan
                stats["stable_median"] = np.nan
                stats["stable_std"] = np.nan
                stats["stable_NMAD"] = np.nan
                stats["stable_RMSE"] = np.nan
                stats["n_stable"] = 0

        self.results.append(stats)

        plt.figure(figsize=(7, 6))
        plt.imshow(dh_arr, cmap="RdBu", vmin=-self.map_clim, vmax=self.map_clim)
        plt.colorbar(label="Elevation change, dh = target - reference (m)")
        plt.title(f"{label} - dDEM")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        sns.histplot(arr, bins=100, kde=True)
        plt.xlabel("Elevation change, dh (m)")
        plt.title(f"{label} - dDEM Distribution")
        plt.tight_layout()
        plt.show()

        slope = xdem.terrain.slope(self.ref, resolution=self.ref.res[0])
        slope_arr = safe_array(slope)

        mask = np.isfinite(dh_arr) & np.isfinite(slope_arr)
        slope_vals = slope_arr[mask].ravel()
        dh_vals = dh_arr[mask].ravel()

        if len(slope_vals) > 0:
            sample = min(50000, len(slope_vals))
            idx = np.random.choice(len(slope_vals), sample, replace=False)

            plt.figure(figsize=(6, 5))
            plt.scatter(slope_vals[idx], dh_vals[idx], alpha=0.2, s=10)
            plt.xlabel("Slope (deg)")
            plt.ylabel("Elevation change, dh (m)")
            plt.title(f"{label} - Slope vs dDEM")
            plt.tight_layout()
            plt.show()

    # ---------------------------------------------------
    # Save final DEM and dDEM products
    # ---------------------------------------------------
    def save_final_outputs(self, pair_name, tgt_raw, tgt_coreg, tgt_corrected):
        dem_dir = output_dir / "final_dem_products"
        dem_dir.mkdir(parents=True, exist_ok=True)

        raw_dh = tgt_raw - self.ref
        coreg_dh = tgt_coreg - self.ref
        corrected_dh = tgt_corrected - self.ref

        safe_pair = pair_name.replace("/", "_").replace("\\", "_").replace(":", "_")

        coreg_dem_path = dem_dir / f"{safe_pair}_target_coregistered.tif"
        corrected_dem_path = (
            dem_dir
            / f"{safe_pair}_target_coreg_bias_aspect_o{self.deramp_order}_a{self.directional_angle}.tif"
        )

        raw_dh_path = dem_dir / f"{safe_pair}_RAW_dDEM.tif"
        coreg_dh_path = dem_dir / f"{safe_pair}_COREG_ONLY_dDEM.tif"
        corrected_dh_path = (
            dem_dir
            / f"{safe_pair}_COREG_BIAS_ASPECT_o{self.deramp_order}_a{self.directional_angle}_dDEM.tif"
        )

        tgt_coreg.save(coreg_dem_path)
        tgt_corrected.save(corrected_dem_path)

        raw_dh.save(raw_dh_path)
        coreg_dh.save(coreg_dh_path)
        corrected_dh.save(corrected_dh_path)

        self.metadata["final_saved_outputs"] = {
            "coregistered_target_dem": str(coreg_dem_path),
            "corrected_target_dem": str(corrected_dem_path),
            "raw_dDEM": str(raw_dh_path),
            "coreg_only_dDEM": str(coreg_dh_path),
            "corrected_dDEM": str(corrected_dh_path),
        }

        print("\n========================================")
        print("FINAL DEM PRODUCTS SAVED")
        print("========================================")
        print(f"Coregistered target DEM: {coreg_dem_path}")
        print(f"Corrected target DEM:    {corrected_dem_path}")
        print(f"Raw dDEM:                {raw_dh_path}")
        print(f"Coreg-only dDEM:         {coreg_dh_path}")
        print(f"Corrected dDEM:          {corrected_dh_path}")

    # ---------------------------------------------------
    # Uncertainty analysis
    # ---------------------------------------------------
    def analyze_uncertainty(self, label, tgt, stable_mask):
        print(f"\nRunning uncertainty estimation for {label}")

        try:
            sig_dh, corr_sig = self.ref.estimate_uncertainty(
                tgt,
                stable_terrain=stable_mask,
                approach="H2022",
                precision_of_other="finer",
                list_vars=("slope", "max_curvature"),
            )

            sig_arr = safe_array(sig_dh)
            finite_sig = sig_arr[np.isfinite(sig_arr)]

            if finite_sig.size == 0:
                mean_unc = np.nan
                median_unc = np.nan
                max_unc = np.nan
                print("No finite uncertainty values returned.")
            else:
                mean_unc = np.mean(finite_sig)
                median_unc = np.median(finite_sig)
                max_unc = np.max(finite_sig)

                print("\n--- Uncertainty Summary ---")
                print(f"Mean sigma_dh:   {mean_unc:.4f} m")
                print(f"Median sigma_dh: {median_unc:.4f} m")
                print(f"Max sigma_dh:    {max_unc:.4f} m")

            plt.figure(figsize=(7, 6))
            plt.imshow(sig_arr, cmap="viridis")
            plt.colorbar(label="Elevation uncertainty, sigma_dh (m)")
            plt.title(f"{label} - Spatial Uncertainty")
            plt.tight_layout()
            plt.show()

            if finite_sig.size > 0:
                plt.figure(figsize=(6, 4))
                sns.histplot(finite_sig, bins=80)
                plt.xlabel("Elevation uncertainty, sigma_dh (m)")
                plt.title(f"{label} - Uncertainty Distribution")
                plt.tight_layout()
                plt.show()

            self.metadata[f"{label}_uncertainty"] = {
                "status": "success",
                "mean_sigma": float(mean_unc),
                "median_sigma": float(median_unc),
                "max_sigma": float(max_unc),
                "spatial_corr_function": corr_sig,
            }

        except Exception as e:
            print("\n--- Uncertainty Summary ---")
            print(f"Uncertainty estimation failed for {label}")
            print(f"Reason: {e}")

            self.metadata[f"{label}_uncertainty"] = {
                "status": "failed",
                "mean_sigma": np.nan,
                "median_sigma": np.nan,
                "max_sigma": np.nan,
                "spatial_corr_function": None,
                "error": str(e),
            }

    # ---------------------------------------------------
    # Production run
    # ---------------------------------------------------
    def run_pair(self, pair_name, tgt):
        print("\n==============================")
        print(f"Processing pair: {pair_name}")
        print("==============================")

        stable_mask = self.build_stable_mask(pair_name, tgt)

        self.analyze(
            label=f"{pair_name}_RAW",
            tgt=tgt,
            stage="raw",
            stable_mask=stable_mask,
        )

        self.analyze_uncertainty(
            label=f"{pair_name}_RAW",
            tgt=tgt,
            stable_mask=stable_mask,
        )

        tgt_coreg = self.coregister(tgt, stable_mask=stable_mask)

        self.analyze(
            label=f"{pair_name}_COREG_ONLY",
            tgt=tgt_coreg,
            stage="coreg",
            stable_mask=stable_mask,
        )

        self.analyze_uncertainty(
            label=f"{pair_name}_COREG_ONLY",
            tgt=tgt_coreg,
            stable_mask=stable_mask,
        )

        tgt_corrected = self.bias_correct_production(
            tgt=tgt_coreg,
            stable_mask=stable_mask,
        )

        if tgt_corrected is None:
            print("Skipping final corrected analysis because correction failed.")
            return

        final_label = (
            f"{pair_name}_COREG_BIAS_ASPECT_o{self.deramp_order}_a{self.directional_angle}"
        )

        self.analyze(
            label=final_label,
            tgt=tgt_corrected,
            stage="coreg_bias_aspect",
            stable_mask=stable_mask,
        )

        self.analyze_uncertainty(
            label=final_label,
            tgt=tgt_corrected,
            stable_mask=stable_mask,
        )

        self.save_final_outputs(
            pair_name=pair_name,
            tgt_raw=tgt,
            tgt_coreg=tgt_coreg,
            tgt_corrected=tgt_corrected,
        )

    # ---------------------------------------------------
    # Summary tables
    # ---------------------------------------------------
    def summarize_runs(self):
        df = pd.DataFrame(self.results).copy()

        if df.empty:
            print("No run results found.")
            return df

        df["abs_median"] = np.abs(df["median"])

        if "stable_median" in df.columns:
            df["stable_abs_median"] = np.abs(df["stable_median"])
        else:
            df["stable_abs_median"] = np.nan

        raw_rows = df[df["stage"] == "raw"]

        if len(raw_rows) > 0:
            raw_row = raw_rows.iloc[0]

            raw_nmad = raw_row.get("stable_NMAD", np.nan)
            raw_rmse = raw_row.get("stable_RMSE", np.nan)
            raw_abs_median = raw_row.get("stable_abs_median", np.nan)

            if np.isfinite(raw_nmad) and raw_nmad != 0:
                df["stable_NMAD_improvement_pct"] = (
                    100 * (raw_nmad - df["stable_NMAD"]) / raw_nmad
                )
            else:
                df["stable_NMAD_improvement_pct"] = np.nan

            if np.isfinite(raw_rmse) and raw_rmse != 0:
                df["stable_RMSE_improvement_pct"] = (
                    100 * (raw_rmse - df["stable_RMSE"]) / raw_rmse
                )
            else:
                df["stable_RMSE_improvement_pct"] = np.nan

            if np.isfinite(raw_abs_median) and raw_abs_median != 0:
                df["stable_bias_improvement_pct"] = (
                    100 * (raw_abs_median - df["stable_abs_median"]) / raw_abs_median
                )
            else:
                df["stable_bias_improvement_pct"] = np.nan

        sort_cols = [
            "stable_NMAD",
            "stable_RMSE",
            "stable_abs_median",
            "NMAD",
            "RMSE",
        ]
        sort_cols = [c for c in sort_cols if c in df.columns]

        rank_df = df.sort_values(sort_cols, ascending=True).reset_index(drop=True)

        if "stable_NMAD" in rank_df.columns:
            plt.figure(figsize=(10, 5))
            sns.barplot(data=rank_df, x="pair", y="stable_NMAD")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Stable-terrain NMAD (m)")
            plt.title("Stable-terrain NMAD by Production Stage")
            plt.tight_layout()
            plt.show()

        return rank_df

    def build_uncertainty_table(self):
        rows = []

        for key, val in self.metadata.items():
            if key.endswith("_uncertainty") and isinstance(val, dict):
                row = {"label": key}
                row.update(val)
                rows.append(row)

        unc_df = pd.DataFrame(rows)

        if not unc_df.empty:
            unc_df = unc_df.sort_values("label").reset_index(drop=True)

        return unc_df

    def info(self):
        print("\nCoregistration + Bias Pipeline Metadata")
        for key, val in self.metadata.items():
            print(f"{key}: {val}")


# ===================================================
# RUN PIPELINE
# ===================================================
pipeline = CoregBiasPipeline(
    reference_label=REFERENCE_LABEL,
    deramp_order=DERAMP_ORDER,
    directional_angle=DIRECTIONAL_ANGLE,
    stable_dh_threshold=STABLE_TERRAIN_DH_THRESHOLD,
    map_clim=MAP_CLIM,
)

pipeline.load_dems(DEM_PATHS, aoi_path=AOI_PATH)

pairs = {
    "post-pre": pipeline.dems["post"],
}

for pair_name, tgt in pairs.items():
    pipeline.run_pair(pair_name, tgt)

pipeline.info()


# ===================================================
# SAVE / PRINT RESULTS
# ===================================================
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:.6f}".format)

df = pd.DataFrame(pipeline.results)
rank_df = pipeline.summarize_runs()
unc_df = pipeline.build_uncertainty_table()

print("\n========================================")
print("PER-RUN METRICS")
print("========================================")
print(df)

print("\n========================================")
print("FINAL RANKED TABLE")
print("========================================")
print(rank_df)

print("\n========================================")
print("UNCERTAINTY SUMMARY TABLE")
print("========================================")
print(unc_df)

output_dir.mkdir(parents=True, exist_ok=True)

raw_csv = output_dir / "xdem_run_metrics_raw.csv"
ranked_csv = output_dir / "xdem_run_metrics_ranked.csv"
unc_csv = output_dir / "xdem_uncertainty_summary.csv"

df.to_csv(raw_csv, index=False)
rank_df.to_csv(ranked_csv, index=False)
unc_df.to_csv(unc_csv, index=False)

print("\n========================================")
print("CSV OUTPUTS SAVED")
print("========================================")
print(f"Raw metrics:         {raw_csv}")
print(f"Ranked metrics:      {ranked_csv}")
print(f"Uncertainty summary: {unc_csv}")
print(f"Histogram values:    {output_dir / 'histogram_values'}")
print(f"Final DEM products:  {output_dir / 'final_dem_products'}")
