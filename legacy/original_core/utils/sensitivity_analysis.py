# utils/sensitivity_analysis.py

# --- Imports (compatible with your pipeline) ---
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

from model.model_builder import ModelBuilder
from SALib.sample import saltelli as saltelli_sample
from SALib.analyze import sobol as sobol_analyze
from joblib import Parallel, delayed
from tqdm import tqdm

# ========= Beamer-ready styling (no LaTeX dependency) =========
_BEAMER_RC = {
    "font.size": 18,  # base size
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.linewidth": 1.4,
    "lines.linewidth": 2.4,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
}


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _default_path(basename: str) -> str:
    """Default save path under ./sensitivity/ when save_path not provided."""
    out_dir = "sensitivity"
    _ensure_dir(out_dir)
    return os.path.join(out_dir, basename)


# ==============================================================
# ===============  COMPUTE: LOCAL SENSITIVITY  =================
# ==============================================================


def compute_local_sensitivity(evolving_traits, bounds, params, epsilon):
    """
    Compute local sensitivity of model parameters with respect to changes in their values.

    Args:
        evolving_traits (List[str]): List of trait names to analyze.
        bounds (List[Tuple[float, float]]): Bounds for each trait.
        params (Dict[str, float]): Model parameters.
        epsilon (float): Perturbation amount for sensitivity analysis.

    Returns:
        Dict[str, float]: Local sensitivity indices for each trait.
    """
    print("\n[INFO] Running Local Sensitivity Analysis...")

    # Base model
    base_model = ModelBuilder(params)
    base_model.build()
    base_results = base_model.run()
    base_fitness = (
        base_results["co2_assimilation_rate_bs"]
        + base_results["co2_assimilation_rate_mes"]
    )

    sensitivities = {}

    for i, trait in enumerate(evolving_traits):
        if trait not in params:
            raise ValueError(f"Trait '{trait}' not found in parameter dictionary.")

        trait_val = params[trait]
        trait_range = bounds[i][1] - bounds[i][0]
        delta = epsilon * trait_range

        perturbed_params = params.copy()
        perturbed_params[trait] = trait_val + delta

        try:
            perturbed_model = ModelBuilder(perturbed_params)
            perturbed_model.build()
            perturbed_results = perturbed_model.run()
            pert_fitness = (
                perturbed_results["co2_assimilation_rate_bs"]
                + perturbed_results["co2_assimilation_rate_mes"]
            )
            sensitivity = abs((pert_fitness - base_fitness) / delta)
            sensitivities[trait] = float(sensitivity)
        except Exception as e:
            print(f"[WARNING] Sensitivity calculation failed for '{trait}': {e}")
            sensitivities[trait] = np.nan

    return sensitivities


# ==============================================================
# ===============   PLOT: LOCAL SENSITIVITY   ==================
# ==============================================================


def plot_local_sensitivity(scores, traits, trait_names, save_path=None):
    """
    Plot local sensitivity analysis results (beamer-ready).
    Saves to ./sensitivity/local_sensitivity.(png|pdf) if save_path not provided.
    """
    # Prepare data
    if isinstance(scores, dict):
        traits = list(scores.keys())
        values = np.array([scores[t] for t in traits], dtype=float)
        names = (
            np.array(trait_names)
            if len(trait_names) == len(traits)
            else np.array(traits)
        )
    else:
        values = np.array(scores, dtype=float)
        names = np.array(trait_names)

    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError("Empty or scalar scores array passed to plotting function.")

    # Sort (descending)
    order = np.argsort(values)[::-1]
    values = values[order]
    names = names[order]

    # Color by log10(values)
    vals_safe = np.where(np.isfinite(values) & (values > 0), values, np.nan)
    log_vals = np.log10(vals_safe)
    vmin, vmax = np.nanmin(log_vals), np.nanmax(log_vals)
    rng = max(vmax - vmin, 1e-12)
    norm = (log_vals - vmin) / rng
    colors = plt.cm.viridis(0.1 + 0.9 * norm)

    with plt.rc_context(_BEAMER_RC):
        fig, ax = plt.subplots(figsize=(11, 6))
        y = np.arange(len(values))
        bars = ax.barh(
            y, values, color=colors, edgecolor="black", linewidth=0.9, height=0.7
        )
        ax.invert_yaxis()

        # Annotate
        for bar, v in zip(bars, values):
            if np.isfinite(v) and v != 0:
                ax.text(
                    bar.get_width() * 1.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2e}",
                    ha="left",
                    va="center",
                    fontsize=14,
                )
            else:
                ax.text(
                    0.02
                    * (np.nanmax(values) if np.isfinite(np.nanmax(values)) else 1.0),
                    bar.get_y() + bar.get_height() / 2,
                    "n/a",
                    ha="left",
                    va="center",
                    fontsize=14,
                    color="0.4",
                )

        ax.set_yticks(y, names)
        ax.set_xlabel("Local sensitivity (Δfitness / Δtrait)", labelpad=8)
        ax.grid(True, which="major", axis="x", alpha=0.25, linewidth=0.8)
        ax.grid(True, which="minor", axis="x", alpha=0.10, linewidth=0.6)
        ax.set_title("Local Sensitivity (sorted)")
        fig.tight_layout()

        # Save (PNG + PDF)
        if save_path is None:
            base = _default_path("local_sensitivity")
        else:
            _ensure_dir(os.path.dirname(save_path) or ".")
            base = os.path.splitext(save_path)[0]

        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
        plt.show()


# ==============================================================
# ==============  COMPUTE: SOBOL SENSITIVITY  ==================
# ==============================================================


def compute_sobol_sensitivity(
    evolving_traits, bounds, params, n_samples=1000, n_jobs=-1
):
    """
    Compute Sobol sensitivity indices for the given model parameters.

    Args:
        evolving_traits (list): List of evolving trait names.
        bounds (list): List of parameter bounds.
        params (dict): Dictionary of model parameters.
        n_samples (int): Number of samples to draw.
        n_jobs (int): Number of parallel jobs to run.

    Returns:
        dict: Sobol sensitivity indices (SALib format).
    """
    print("\n[INFO] Running Global (Sobol) Sensitivity Analysis...")

    problem = {
        "num_vars": len(evolving_traits),
        "names": evolving_traits,
        "bounds": bounds,
    }

    param_values = saltelli_sample.sample(problem, N=n_samples, calc_second_order=True)
    failed_samples = []

    def evaluate_model(x):
        try:
            perturbed_params = params.copy()
            trait_dict = {
                name: float(np.clip(val, *bnd))
                for (name, val, bnd) in zip(evolving_traits, x, bounds)
            }
            if "g_bs" in trait_dict:
                trait_dict["g_bs"] = max(trait_dict["g_bs"], 1e-5)

            perturbed_params.update(trait_dict)
            model = ModelBuilder(perturbed_params)
            model.build()
            for name, val in trait_dict.items():
                model.set_parameter_value(name, val)

            result = model.run()
            fitness = (
                result["co2_assimilation_rate_bs"] + result["co2_assimilation_rate_mes"]
            )
            return True, float(fitness)
        except Exception as e:
            print(f"[WARN] Sobol evaluation failed for x={x.tolist()} with error: {e}")
            failed_samples.append(x.tolist())
            return False, np.nan

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_model)(x)
        for x in tqdm(param_values, desc="[INFO] Evaluating Sobol samples")
    )

    success_flags, outputs = zip(*results)
    outputs = np.array(outputs, dtype=float)

    success_count = int(np.sum(success_flags))
    failure_count = len(success_flags) - success_count
    print(
        f"[INFO] Sobol run completed: {success_count} succeeded, {failure_count} failed of {len(success_flags)}."
    )

    if np.any(np.isnan(outputs)):
        n_bad = int(np.isnan(outputs).sum())
        print(
            f"[INFO] {n_bad} NaN outputs detected. Replacing with mean of valid results."
        )
        if np.all(np.isnan(outputs)):
            raise RuntimeError(
                "All Sobol samples failed. Cannot proceed with analysis."
            )
        outputs[np.isnan(outputs)] = np.nanmean(outputs)

    if failed_samples:
        print(
            "[INFO] Logging failed parameter combinations to failed_sobol_samples.csv"
        )
        pd.DataFrame(failed_samples, columns=evolving_traits).to_csv(
            "failed_sobol_samples.csv", index=False
        )

    return sobol_analyze.analyze(
        problem, outputs, calc_second_order=True, print_to_console=False
    )


# ==============================================================
# ==============    PLOT: SOBOL S1 / ST / S2 SEPARATE ==========
# ==============================================================


def plot_sobol_sensitivity(sobol_result, traits, figsize=(11, 6), save_path=None):
    """
    Save three SEPARATE plots (S1, ST, S2) for beamer.
      - If save_path is None -> ./sensitivity/sobol_S1.*, sobol_ST.*, sobol_S2.*
      - If save_path is "path/base.ext" -> path/base_S1.*, base_ST.*, base_S2.*
    S2 shows only the UPPER TRIANGLE (pair-plot style). Diagonal kept.
    """
    if not isinstance(sobol_result, dict):
        raise ValueError("Input must be a dictionary.")
    for key in ["S1", "ST"]:
        if key not in sobol_result:
            raise ValueError(f"Missing key '{key}' in sobol_result.")

    names = list(traits)
    n = len(names)
    S1 = np.asarray(sobol_result["S1"], dtype=float)
    ST = np.asarray(sobol_result["ST"], dtype=float)
    S1_conf = np.asarray(sobol_result.get("S1_conf", np.zeros_like(S1)), dtype=float)
    ST_conf = np.asarray(sobol_result.get("ST_conf", np.zeros_like(ST)), dtype=float)
    S2 = sobol_result.get("S2", None)
    has_S2 = isinstance(S2, (list, np.ndarray)) and np.asarray(S2).shape == (n, n)
    if has_S2:
        S2 = np.asarray(S2, dtype=float)

    # Base path handling
    if save_path is None:
        base = _default_path("sobol")
    else:
        _ensure_dir(os.path.dirname(save_path) or ".")
        base = os.path.splitext(save_path)[0]

    # ----- S1 (separate figure) -----
    with plt.rc_context(_BEAMER_RC):
        fig, ax = plt.subplots(figsize=figsize)
        order = np.argsort(S1)
        y = np.arange(n)
        bars = ax.barh(
            y,
            S1[order],
            xerr=S1_conf[order],
            color="#377eb8",
            edgecolor="black",
            linewidth=0.8,
            height=0.7,
            ecolor="0.2",
            capsize=4,
        )
        ax.invert_yaxis()
        ax.set_yticks(y, np.array(names)[order])
        ax.set_xlabel(r"First-order index ($S_1$)", labelpad=8)
        # ax.set_title("Sobol — First-order (S₁)")
        ax.grid(True, axis="x", alpha=0.25)
        for bar, v in zip(bars, S1[order]):
            ax.text(
                v + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}",
                ha="left",
                va="center",
                fontsize=14,
            )
        fig.tight_layout()
        fig.savefig(base + "_S1.png", dpi=300, bbox_inches="tight")
        fig.savefig(base + "_S1.pdf", dpi=300, bbox_inches="tight")
        plt.show()

    # ----- ST (separate figure) -----
    with plt.rc_context(_BEAMER_RC):
        fig, ax = plt.subplots(figsize=figsize)
        order = np.argsort(ST)
        y = np.arange(n)
        bars = ax.barh(
            y,
            ST[order],
            xerr=ST_conf[order],
            color="#e41a1c",
            edgecolor="black",
            linewidth=0.8,
            height=0.7,
            ecolor="0.2",
            capsize=4,
        )
        ax.invert_yaxis()
        ax.set_yticks(y, np.array(names)[order])
        ax.set_xlabel(r"Total-order index ($S_T$)", labelpad=8)
        # ax.set_title("Sobol — Total-order (Sₜ)")
        ax.grid(True, axis="x", alpha=0.25)
        for bar, v in zip(bars, ST[order]):
            ax.text(
                v + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}",
                ha="left",
                va="center",
                fontsize=14,
            )
        fig.tight_layout()
        fig.savefig(base + "_ST.png", dpi=300, bbox_inches="tight")
        fig.savefig(base + "_ST.pdf", dpi=300, bbox_inches="tight")
        plt.show()

    # ----- S2 (separate figure, upper triangle only) -----
    if has_S2:
        with plt.rc_context(_BEAMER_RC):
            fig, ax = plt.subplots(figsize=(11, 8))

            # Mask lower triangle -> show upper like pair-plot
            mask = np.tril(
                np.ones_like(S2, dtype=bool), k=-1
            )  # True on lower off-diagonal
            # To hide diagonal too, uncomment:
            # mask |= np.eye(n, dtype=bool)

            vmax = np.nanmax(np.abs(S2)) if np.isfinite(S2).any() else 1.0
            sns.heatmap(
                S2,
                ax=ax,
                mask=mask,
                cmap="RdBu_r",
                center=0,
                vmin=-vmax,
                vmax=vmax,
                annot=True,
                fmt=".2f",
                cbar_kws={"shrink": 0.75},
                linewidths=0.5,
                linecolor="white",
            )
            ax.set_xticks(np.arange(n) + 0.5)
            ax.set_yticks(np.arange(n) + 0.5)
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticklabels(names, rotation=0)
            # ax.set_title("Second-order interactions (S₂)")
            cbar = ax.collections[0].colorbar
            cbar.set_label("Interaction strength", fontsize=16)

            fig.tight_layout()
            fig.savefig(base + "_S2.png", dpi=300, bbox_inches="tight")
            fig.savefig(base + "_S2.pdf", dpi=300, bbox_inches="tight")
            plt.show()


# ==============================================================
# ======  PLOT: SOBOL (THESIS STYLE) — SEPARATE FILES  =========
# ==============================================================


def plot_sobol_sensitivity_thesis(
    sobol_result, traits, trait_names, figsize=(11, 6), save_path=None
):
    """
    Publication/Beamer-ready: writes three separate figures:
      *_thesis_S1.*, *_thesis_ST.*, *_thesis_S2.* (upper triangle only if S2 present).
    """
    n = len(traits)
    if len(trait_names) != n:
        raise ValueError("Length of trait_names must match length of traits.")

    S1 = np.asarray(sobol_result.get("S1"), dtype=float)
    ST = np.asarray(sobol_result.get("ST"), dtype=float)
    S2 = sobol_result.get("S2", None)
    has_S2 = isinstance(S2, (list, np.ndarray)) and np.asarray(S2).shape == (n, n)
    if has_S2:
        S2 = np.asarray(S2, dtype=float)

    # base path
    if save_path is None:
        base = _default_path("sobol_thesis")
    else:
        _ensure_dir(os.path.dirname(save_path) or ".")
        base = os.path.splitext(save_path)[0]

    # S1
    with plt.rc_context(_BEAMER_RC):
        fig, ax = plt.subplots(figsize=figsize)
        order = np.argsort(S1)
        y = np.arange(n)
        ax.barh(
            y, S1[order], color="#377eb8", edgecolor="black", linewidth=0.8, height=0.7
        )
        ax.invert_yaxis()
        ax.set_yticks(y, np.array(trait_names)[order])
        ax.set_xlabel(r"First-order index ($S_1$)", labelpad=8)
        # ax.set_title("S₁")
        ax.grid(True, axis="x", alpha=0.25)
        for yi, v in zip(y, S1[order]):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", ha="left", fontsize=14)
        fig.tight_layout()
        fig.savefig(base + "_S1.png", dpi=300, bbox_inches="tight")
        fig.savefig(base + "_S1.pdf", dpi=300, bbox_inches="tight")
        plt.show()

    # ST
    with plt.rc_context(_BEAMER_RC):
        fig, ax = plt.subplots(figsize=figsize)
        order = np.argsort(ST)
        y = np.arange(n)
        ax.barh(
            y, ST[order], color="#2ca02c", edgecolor="black", linewidth=0.8, height=0.7
        )
        ax.invert_yaxis()
        ax.set_yticks(y, np.array(trait_names)[order])
        ax.set_xlabel(r"Total-order index ($S_T$)", labelpad=8)
        # ax.set_title("Sₜ")
        ax.grid(True, axis="x", alpha=0.25)
        for yi, v in zip(y, ST[order]):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", ha="left", fontsize=14)
        fig.tight_layout()
        fig.savefig(base + "_ST.png", dpi=300, bbox_inches="tight")
        fig.savefig(base + "_ST.pdf", dpi=300, bbox_inches="tight")
        plt.show()

    # S2 (upper triangle only)
    if has_S2:
        with plt.rc_context(_BEAMER_RC):
            fig, ax = plt.subplots(figsize=(11, 8))
            mask = np.tril(np.ones_like(S2, dtype=bool), k=-1)
            vmax = np.nanmax(np.abs(S2)) if np.isfinite(S2).any() else 1.0
            sns.heatmap(
                S2,
                ax=ax,
                mask=mask,
                cmap="RdBu_r",
                center=0,
                vmin=-vmax,
                vmax=vmax,
                annot=True,
                fmt=".3f",
                cbar_kws={"shrink": 0.75},
                linewidths=0.5,
                linecolor="white",
            )
            ax.set_xticks(np.arange(n) + 0.5)
            ax.set_yticks(np.arange(n) + 0.5)
            ax.set_xticklabels(trait_names, rotation=45, ha="right")
            ax.set_yticklabels(trait_names, rotation=0)
            # ax.set_title("Second-order interactions")
            cbar = ax.collections[0].colorbar
            cbar.set_label("Interaction strength", fontsize=16)
            fig.tight_layout()
            fig.savefig(base + "_S2.png", dpi=300, bbox_inches="tight")
            fig.savefig(base + "_S2.pdf", dpi=300, bbox_inches="tight")
            plt.show()
