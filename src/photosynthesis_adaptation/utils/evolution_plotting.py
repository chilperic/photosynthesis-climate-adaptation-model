# ==============================================================
#  Publication-ready evolutionary visualization suite (enzyme-aware)
#  Author: Dr. Chilperic Armel Foko Kuate
#  Rev: 2025-10-29 — Enzyme-fraction + δ13C + Fitness Evolution Maps
# ==============================================================

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d


# --------------------------------------------------------------
#  Global RC defaults
# --------------------------------------------------------------


_PRESENTATION_RC = {
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
    "axes.linewidth": 1.3,
    "lines.linewidth": 2.4,
    "lines.markersize": 7,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "figure.dpi": 160,
    "savefig.dpi": 350,
}


# ==============================================================
#  Publication-ready evolution visualization — Data sanitation patch
# ==============================================================


def _sanitize_history(history):
    """
    Cleans invalid / negative assimilation values in the history dictionary.
    Ensures color scaling and plotting are consistent across functions.
    """
    clean = dict(history)

    # --- Individual-level arrays ---
    for key in ["fitness", "fitness_bs", "fitness_mes"]:
        if key in clean:
            arr = np.asarray(clean[key], float)
            arr[~np.isfinite(arr)] = np.nan
            arr[arr < 0] = np.nan  # clip negatives
            clean[key] = arr.tolist()

    # --- Generation-level means ---
    if "mean_fitness_per_gen" in clean:
        arr = np.asarray(clean["mean_fitness_per_gen"], float)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr < 0] = np.nan
        clean["mean_fitness_per_gen"] = arr.tolist()

    # --- Mean traits: remove NaN and enforce presence of kinetic terms ---
    if "mean_traits_per_gen" in clean:
        cleaned_traits = []
        for t in clean["mean_traits_per_gen"]:
            t = {k: (v if np.isfinite(v) else np.nan) for k, v in t.items()}
            for k in ["v_cmax_mes", "v_cmax_bs", "v_pmax"]:
                t.setdefault(k, np.nan)
            cleaned_traits.append(t)
        clean["mean_traits_per_gen"] = cleaned_traits

    return clean


def _prepare_color_norm(values, cmap_name="plasma"):
    """Utility: build cmap + norm with lower bound fixed at zero."""
    arr = np.asarray(values, float)
    arr = np.clip(arr, 0, np.nanmax(arr))
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=np.nanmax(arr))
    return cmap, norm


def _mask_valid(*arrays):
    """Returns mask of finite & positive entries for multiple arrays."""
    mask = np.ones_like(np.asarray(arrays[0], float), dtype=bool)
    for a in arrays:
        a = np.asarray(a, float)
        mask &= np.isfinite(a) & (a >= 0)
    return mask


def _add_horizontal_colorbar(fig, mappable, label=None, height=0.03, bottom=0.10):
    """
    Adds a clean, centered horizontal colorbar below all subplots.
    - fig: current matplotlib figure
    - mappable: the ScalarMappable (e.g. scatter, cm.ScalarMappable)
    - label: optional colorbar label
    - height: colorbar thickness (fraction of figure height)
    - bottom: vertical position (fraction of figure height)
    """
    cax = fig.add_axes([0.15, bottom, 0.70, height])  # [left, bottom, width, height]
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    if label:
        cbar.set_label(label, fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    return cbar


def _add_colorbar(sc_or_cmap, fig, axes, orientation="horizontal", label=None):
    """
    Adds a consistent publication-style colorbar to any figure.
    - sc_or_cmap: either a ScalarMappable (scatter handle) or cmap+norm pair.
    - fig: current figure object.
    - axes: one or more Axes objects.
    """
    cbar = fig.colorbar(
        sc_or_cmap if hasattr(sc_or_cmap, "get_array") else sc_or_cmap,
        ax=axes,
        orientation=orientation,
        shrink=0.8,
        pad=0.08 if orientation == "vertical" else 0.25,
        aspect=30,
    )

    if orientation == "horizontal":
        cbar.ax.xaxis.set_ticks_position("bottom")
        cbar.ax.xaxis.set_label_position("bottom")

    if label:
        cbar.set_label(label, fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    return cbar


def _rc(extra=None):
    rc = dict(_PRESENTATION_RC)
    if extra:
        rc.update(extra)
    return plt.rc_context(rc)


def _apply_masked_array(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


# ==============================================================
#  δ13C trajectory — interpretable physiological zones
# ==============================================================


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plot_delta13C_trajectory_clean(
    delta13c_data, generations=None, save_path=None, smooth_sigma=0.05, annotate=True
):
    """δ13C trajectory with shaded C₃–C₄ transition zones and standard deviation around the trajectory for the entire population."""

    # Convert the data to a numpy array
    d13c = np.asarray(delta13c_data, float)

    # If generations are not provided, create a range of indices
    gens = np.arange(len(d13c)) if generations is None else np.asarray(generations)

    # Mask for valid δ13C values within reasonable bounds
    mask = np.isfinite(d13c) & (d13c > -40) & (d13c < 0)
    d13c, gens = d13c[mask], gens[mask]

    # If there are too few data points, exit early
    if len(d13c) < 3:
        print("⚠️ Not enough δ13C data to plot.")
        return

    # Calculate the standard deviation for each generation
    unique_gens = np.unique(gens)
    mean_values = [np.mean(d13c[gens == g]) for g in unique_gens]
    std_values = [np.std(d13c[gens == g]) for g in unique_gens]
    min_values = [np.min(d13c[gens == g]) for g in unique_gens]
    max_values = [np.max(d13c[gens == g]) for g in unique_gens]

    # Define the C₃, intermediate, and C₄ transition zones
    c3_zone, interm_zone, c4_zone = (-30, -24), (-24, -16), (-16, -10)

    # Plot the data
    with _rc():  # Applying RC for presentation style
        fig, ax = plt.subplots(figsize=(9, 5.5))

        # Background shading for C₃, intermediate, and C₄ zones
        ax.axhspan(*c3_zone, color="#8B0000", alpha=0.08, label="C₃ zone")
        ax.axhspan(
            *interm_zone, color="#FFD700", alpha=0.08, label="C₃–C₄ intermediate"
        )
        ax.axhspan(*c4_zone, color="#006400", alpha=0.08, label="C₄ zone")

        # Add lines for the C₃ to C₄ transition
        ax.axhline(-27, color="#8B0000", ls="--", lw=1.8)  # C₃ line
        ax.axhline(-13, color="#006400", ls="--", lw=1.8)  # C₄ line

        # Plot the δ13C trajectory as individual points for each generation
        for gen in unique_gens:
            # Filter data for the current generation
            gen_values = d13c[gens == gen]

            # Scatter plot for each generation's population values
            ax.scatter(
                [gen] * len(gen_values), gen_values, color="#1f77b4", alpha=0.4, s=20
            )

        # Add a line for the mean δ13C value per generation
        ax.plot(
            unique_gens,
            mean_values,
            "-o",
            color="#1f77b4",
            lw=2.5,
            label=r"$\delta^{13}\mathrm{C}$ trajectory (mean)",
        )

        # Add shaded area for the standard deviation around the mean
        ax.fill_between(
            unique_gens,
            np.array(mean_values) - np.array(std_values),
            np.array(mean_values) + np.array(std_values),
            color="#1f77b4",
            alpha=0.2,
        )

        # Add lines for the min and max δ13C values per generation (optional, can be removed if not needed)
        ax.fill_between(unique_gens, min_values, max_values, color="#1f77b4", alpha=0.1)

        # Set axis labels
        ax.set_xlabel("Generation", labelpad=10)
        ax.set_ylabel(r"$\delta^{13}\mathrm{C}$ (‰)", labelpad=10)
        ax.set_ylim(-30, -8)
        ax.grid(True, ls="--", lw=0.4, alpha=0.4)

        # Annotate with the range of δ13C values
        if annotate:
            min_val, max_val = np.nanmin(mean_values), np.nanmax(mean_values)
            ax.text(
                0.02,
                0.95,
                f"Range: {min_val:.1f} → {max_val:.1f}‰",
                transform=ax.transAxes,
                fontsize=12.5,
                ha="left",
                va="top",
                color="dimgray",
            )

            # Annotate C₃ to C₄ transitions
            for i in range(1, len(mean_values)):
                if (mean_values[i - 1] < -16) and (mean_values[i] >= -16):
                    ax.axvline(unique_gens[i], color="gray", ls=":", lw=1)
                    ax.text(
                        unique_gens[i],
                        -29,
                        "C₄ entry",
                        rotation=90,
                        va="bottom",
                        ha="center",
                        fontsize=11,
                        color="gray",
                    )
                elif (mean_values[i - 1] < -24) and (mean_values[i] >= -24):
                    ax.axvline(unique_gens[i], color="gray", ls=":", lw=1)
                    ax.text(
                        unique_gens[i],
                        -29,
                        "Intermediate",
                        rotation=90,
                        va="bottom",
                        ha="center",
                        fontsize=11,
                        color="gray",
                    )

        # Show the legend
        ax.legend(frameon=False, loc="lower right", handlelength=2.5)
        plt.subplots_adjust(bottom=0.15, left=0.15)

        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()


# ==============================================================
#  Trait dynamics
# ==============================================================


def plot_grouped_trait_dynamics(
    trait_history,
    traits,
    trait_names,
    generations=None,
    save_path=None,
    n_cols=2,
    figsize_per_col=(6.0, 3.8),
):
    """Plots grouped trait trajectories (enzyme fractions, conductances, etc.)."""
    n_plots = len(traits)
    n_rows = math.ceil(n_plots / n_cols)
    gens = (
        np.arange(len(trait_history))
        if generations is None
        else np.asarray(generations)
    )

    with _rc():
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figsize_per_col[0] * n_cols, figsize_per_col[1] * n_rows),
            squeeze=False,
        )
        for i, key in enumerate(traits):
            row, col = divmod(i, n_cols)
            vals = gaussian_filter1d(
                np.array([d.get(key, np.nan) for d in trait_history]), sigma=1
            )
            ax = axes[row][col]
            ax.plot(gens, vals, "-o", lw=2.2, ms=4.5)
            ax.set_xlabel("Generation")
            ax.set_ylabel(trait_names[i])
            ax.grid(True, alpha=0.4)
            if "f_pepc" in key:
                ax.set_facecolor("#FFF4EC")
            elif "f_rubisco_mes" in key:
                ax.set_facecolor("#ECF7FF")
            elif "f_rubisco_bs" in key:
                ax.set_facecolor("#EEFFEE")
        for j in range(n_plots, n_rows * n_cols):
            fig.delaxes(axes.flatten()[j])
        fig.tight_layout(pad=1.0)
        plt.subplots_adjust(hspace=0.35, wspace=0.28)
        if save_path:
            plt.savefig(save_path, dpi=350, bbox_inches="tight")
        plt.show()


# ==============================================================
#  Improved 3D evolutionary trajectory (normalized)
# ==============================================================


def plot_3d_evolution_trajectory(
    trait_history, fitness_history, save_path=None, cmap_name="plasma"
):
    """3D trajectory showing *real* assimilation (μmol m⁻² s⁻¹)."""
    f_pepc = np.array([d.get("f_pepc", np.nan) for d in trait_history])
    f_bs = np.array([d.get("f_rubisco_bs", np.nan) for d in trait_history])
    f_mes = np.array([d.get("f_rubisco_mes", np.nan) for d in trait_history])
    A = np.array(fitness_history)

    mask = np.isfinite(f_pepc) & np.isfinite(f_bs) & np.isfinite(f_mes) & np.isfinite(A)
    f_pepc, f_bs, f_mes, A = f_pepc[mask], f_bs[mask], f_mes[mask], A[mask]
    if len(A) < 3:
        print("⚠️ Not enough valid points to plot 3D trajectory.")
        return

    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=np.nanmin(A), vmax=np.nanmax(A))

    with _rc():
        fig = plt.figure(figsize=(8, 6.5))
        ax = fig.add_subplot(111, projection="3d")

        for i in range(len(A) - 1):
            ax.plot(
                f_pepc[i : i + 2],
                f_bs[i : i + 2],
                f_mes[i : i + 2],
                color=cmap(norm(A[i])),
                lw=2.5,
            )

        ax.scatter(
            f_pepc[0],
            f_bs[0],
            f_mes[0],
            color="crimson",
            s=120,
            marker="X",
            label="Start",
        )
        ax.scatter(
            f_pepc[-1],
            f_bs[-1],
            f_mes[-1],
            color="lime",
            s=140,
            marker="*",
            label="End",
        )

        ax.set_xlabel("fₚₑₚc (PEPC fraction)")
        ax.set_ylabel("fᵣᵤbisco,BS")
        ax.set_zlabel("fᵣᵤbisco,Mes")
        ax.legend(frameon=False, loc="upper left")

        _add_horizontal_colorbar(
            fig,
            cm.ScalarMappable(norm=norm, cmap=cmap),
            label="Assimilation (μmol m⁻² s⁻¹)",
            bottom=0.03,
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()


# ==============================================================
#  2D Fraction–Fitness evolutionary map
# ==============================================================


def plot_fraction_fitness_map(
    trait_history, fitness_history, save_path=None, cmap_name="plasma"
):
    """2D enzyme fraction map with color = fitness and arrows = evolutionary direction."""
    f_pepc = np.array([d.get("f_pepc", np.nan) for d in trait_history])
    f_bs = np.array([d.get("f_rubisco_bs", np.nan) for d in trait_history])
    A = np.array(fitness_history)
    mask = np.isfinite(f_pepc) & np.isfinite(f_bs) & np.isfinite(A)
    f_pepc, f_bs, A = f_pepc[mask], f_bs[mask], A[mask]

    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=np.nanmin(A), vmax=np.nanmax(A))

    with _rc():
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        sc = ax.scatter(
            f_pepc, f_bs, c=A, cmap=cmap, norm=norm, s=60, edgecolor="k", lw=0.3
        )
        ax.plot(f_pepc, f_bs, "-", color="gray", alpha=0.4)
        for i in range(0, len(f_pepc) - 1, max(1, len(f_pepc) // 12)):
            ax.arrow(
                f_pepc[i],
                f_bs[i],
                f_pepc[i + 1] - f_pepc[i],
                f_bs[i + 1] - f_bs[i],
                color="gray",
                alpha=0.7,
                lw=0.8,
                head_width=0.002,
                length_includes_head=True,
            )
        ax.scatter(
            f_pepc[0], f_bs[0], s=120, marker="X", color="crimson", label="Start (C₃)"
        )
        ax.scatter(
            f_pepc[-1], f_bs[-1], s=140, marker="*", color="lime", label="End (C₄)"
        )
        ax.set_xlabel(r"$f_{\mathrm{PEPC}}$")
        ax.set_ylabel(r"$f_{\mathrm{Rubisco,BS}}$")
        ax.legend(frameon=False)
        _add_horizontal_colorbar(
            fig,
            sc,
            label="Assimilation rate (μmol m⁻² s⁻¹)",
            bottom=0.02,
        )
        fig.tight_layout(rect=[0, 0.10, 1, 0.97])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()


# ==============================================================
#  Pareto front
# ==============================================================


def plot_pareto_front(fitness_bs, fitness_mes, save_path=None):
    """Pareto trade-off between BS and mesophyll assimilation."""
    bs, mes = _apply_masked_array(fitness_bs), _apply_masked_array(fitness_mes)
    tot = bs + mes
    cmap = cm.get_cmap("plasma")
    norm = mpl.colors.Normalize(vmin=np.nanmin(tot), vmax=np.nanmax(tot))
    with _rc():
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            bs,
            mes,
            c=tot,
            cmap=cmap,
            norm=norm,
            edgecolors="0.2",
            linewidths=0.4,
            s=50,
            alpha=0.9,
        )
        ax.set_xlabel(r"A$_\mathrm{bs}$ ($\mu$mol m$^{-2}$ s$^{-1}$)", labelpad=10)
        ax.set_ylabel(r"A$_\mathrm{mes}$ ($\mu$mol m$^{-2}$ s$^{-1}$)", labelpad=10)
        ax.grid(True, ls="--", lw=0.4, alpha=0.4)
        _add_horizontal_colorbar(
            fig,
            sc,
            label="Assimilation rate (μmol m⁻² s⁻¹)",
            bottom=0.010,
        )

        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "--", color="gray", alpha=0.5, lw=1)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=350, bbox_inches="tight")
        plt.show()


# ==============================================================
#  Comprehensive evolution overview
# ==============================================================


def plot_evolution_overview(
    history, save_path=None, cmap_name="plasma", smooth_sigma=0.01, arrow_every=1
):
    """Overview: Color = fitness, Size = f_Rubisco,Mes, Arrows = generations."""
    mean_traits = history["mean_traits_per_gen"]
    fitness = np.asarray(history["mean_fitness_per_gen"], float)
    gens = np.arange(len(mean_traits))

    f_pepc = gaussian_filter1d(
        [t.get("f_pepc", np.nan) for t in mean_traits], sigma=smooth_sigma
    )
    f_bs = gaussian_filter1d(
        [t.get("f_rubisco_bs", np.nan) for t in mean_traits], sigma=smooth_sigma
    )
    f_mes = gaussian_filter1d(
        [t.get("f_rubisco_mes", np.nan) for t in mean_traits], sigma=smooth_sigma
    )
    mask = (
        np.isfinite(f_pepc)
        & np.isfinite(f_bs)
        & np.isfinite(f_mes)
        & np.isfinite(fitness)
    )
    f_pepc, f_bs, f_mes, fitness, gens = (
        f_pepc[mask],
        f_bs[mask],
        f_mes[mask],
        fitness[mask],
        gens[mask],
    )

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=np.nanmin(fitness), vmax=np.nanmax(fitness))
    size_scaled = 100 + 500 * (f_mes - np.nanmin(f_mes)) / (
        np.nanmax(f_mes) - np.nanmin(f_mes)
    )

    with _rc():
        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        sc = ax.scatter(
            f_pepc,
            f_bs,
            c=fitness,
            s=size_scaled,
            cmap=cmap,
            norm=norm,
            edgecolor="k",
            lw=0.3,
            alpha=0.9,
            zorder=3,
        )
        ax.plot(f_pepc, f_bs, "-", color="gray", alpha=0.5, lw=1.4, zorder=1)
        for i in range(0, len(f_pepc) - 1, arrow_every):
            ax.arrow(
                f_pepc[i],
                f_bs[i],
                f_pepc[i + 1] - f_pepc[i],
                f_bs[i + 1] - f_bs[i],
                color="black",
                alpha=0.01,
                lw=0,
                head_width=0.012 * np.ptp(f_pepc),
                head_length=0.02 * np.ptp(f_bs),
                length_includes_head=True,
                zorder=4,
            )
        ax.scatter(
            f_pepc[0],
            f_bs[0],
            color="crimson",
            s=220,
            marker="X",
            edgecolor="k",
            label="Start (C₃-like)",
        )
        ax.scatter(
            f_pepc[-1],
            f_bs[-1],
            color="lime",
            s=260,
            marker="*",
            edgecolor="k",
            label="End (C₄-like)",
        )
        ax.set_xlabel(r"$f_\mathrm{PEPC}$ (PEPC fraction)", labelpad=10)
        ax.set_ylabel(r"$f_\mathrm{Rubisco,BS}$ (BS Rubisco fraction)", labelpad=10)
        ax.grid(True, ls="--", lw=0.5, alpha=0.6)
        ax.legend(frameon=False, loc="upper left")
        _add_colorbar(
            sc,
            fig,
            ax,
            orientation="horizontal",
            label="Assimilation rate (μmol m$^{-2}$ s$^{-1}$)",
        )

        for frac, s in zip([0.3, 0.6, 0.9], [150, 350, 550]):
            ax.scatter(
                [],
                [],
                s=s,
                color="gray",
                alpha=0.4,
                label=f"$f_{{Rubisco,Mes}}$ ≈ {frac:.1f}",
            )
        ax.legend(frameon=False, loc="best", fontsize=10)
        ax.text(
            0.5,
            -0.3,
            "Evolutionary map: Color = Assimilation | Size = f_Rubisco,Mes | Arrows = Generations",
            ha="center",
            fontsize=12,
            color="dimgray",
            transform=ax.transAxes,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()
# ==============================================================
#  2D evolutionary direction (with assimilation encoding)
# ==============================================================


def plot_evolution_direction(
    history,
    trait_x="v_cmax_mes",
    trait_y="v_cmax_bs",
    save_path=None,
    cmap_name="plasma",
    smooth_sigma=1.0,
    arrow_every=3,
    show_generations=True,
):
    """
    2D projection of mean evolution path across generations.
    Color = assimilation rate (fitness), Size = generation.
    """
    mean_traits = history.get("mean_traits_per_gen", [])
    fitness = np.asarray(history.get("mean_fitness_per_gen", []), dtype=float)

    if len(mean_traits) == 0 or len(fitness) == 0:
        print("⚠️ No valid data in history for plotting.")
        return

    gens = np.arange(len(mean_traits))

    # Extract and smooth traits
    x = gaussian_filter1d(
        np.array([t.get(trait_x, np.nan) for t in mean_traits], dtype=float),
        sigma=smooth_sigma,
    )
    y = gaussian_filter1d(
        np.array([t.get(trait_y, np.nan) for t in mean_traits], dtype=float),
        sigma=smooth_sigma,
    )

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(fitness)
    if not np.any(mask):
        print(f"⚠️ No valid (finite) values found for {trait_x} vs {trait_y}.")
        return

    x, y, fitness, gens = x[mask], y[mask], fitness[mask], gens[mask]

    # Handle degenerate cases
    if len(x) < 3:
        print("⚠️ Not enough valid generations to plot.")
        return

    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=np.nanmin(fitness), vmax=np.nanmax(fitness))
    colors = cmap(norm(fitness))
    sizes = 80 + 300 * (gens / np.nanmax(gens))

    with _rc():
        fig, ax = plt.subplots(figsize=(8.0, 7.0))
        sc = ax.scatter(
            x,
            y,
            c=fitness,
            cmap=cmap,
            norm=norm,
            s=sizes,
            edgecolor="k",
            lw=0.3,
            zorder=3,
            alpha=0.9,
        )
        ax.plot(x, y, "-", color="gray", alpha=0.4, lw=1.4, zorder=1)

        for i in range(0, len(x) - 1, arrow_every):
            dx, dy = x[i + 1] - x[i], y[i + 1] - y[i]
            ax.arrow(
                x[i],
                y[i],
                dx,
                dy,
                color=cmap(norm(fitness[i])),
                alpha=0.8,
                lw=0,
                head_width=0.015 * np.ptp(x),
                head_length=0.02 * np.ptp(y),
                length_includes_head=True,
                zorder=4,
            )

        ax.scatter(
            x[0],
            y[0],
            color="crimson",
            s=200,
            marker="X",
            edgecolor="k",
            label="Start (C₃-like)",
        )
        ax.scatter(
            x[-1],
            y[-1],
            color="lime",
            s=240,
            marker="*",
            edgecolor="k",
            label="End (C₄-like)",
        )

        if show_generations:
            for i in range(0, len(x), max(1, len(x) // 8)):
                ax.text(
                    x[i],
                    y[i],
                    f"{gens[i]}",
                    fontsize=10,
                    color="dimgray",
                    ha="center",
                    va="center",
                    zorder=5,
                    alpha=0.7,
                )

        ax.set_xlabel(trait_x.replace("_", " ").title(), labelpad=10)
        ax.set_ylabel(trait_y.replace("_", " ").title(), labelpad=10)
        ax.grid(True, ls="--", lw=0.5, alpha=0.6)
        ax.legend(frameon=False, loc="upper left")

        _add_colorbar(
            sc,
            fig,
            ax,
            orientation="horizontal",
            label="Assimilation rate (μmol m$^{-2}$ s$^{-1}$)",
        )


        ax.text(
            0.5,
            -0.12,
            "Color = Assimilation | Size = Generation | Arrows = Evolution direction",
            ha="center",
            fontsize=12,
            color="dimgray",
            transform=ax.transAxes,
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()


# ==============================================================
#  Evolution visualization suite — 3 complementary modes
#  Author: Dr. Chilperic Armel Foko Kuate
#  Date: 2025-10-29
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


# --------------------------------------------------------------
#  Utility functions
# --------------------------------------------------------------


def _rc_context():
    return plt.rc_context(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0,
            "figure.dpi": 160,
            "savefig.dpi": 350,
        }
    )


def _get_valid_arrays(history, keys):
    mean_traits = history.get("mean_traits_per_gen", [])
    if not mean_traits:
        raise ValueError("Missing mean_traits_per_gen in history.")
    arrs = [np.array([t.get(k, np.nan) for t in mean_traits], float) for k in keys]
    fitness = np.asarray(history.get("mean_fitness_per_gen", []), float)
    gens = np.arange(len(fitness))
    mask = np.isfinite(fitness)
    for a in arrs:
        mask &= np.isfinite(a)
    return [a[mask] for a in arrs], fitness[mask], gens[mask]


# ==============================================================
#  OPTION 1 — Clean 2D projections (Publication-ready)
# ==============================================================


def plot_evolution_summary(history, save_path=None, cmap_name="plasma"):
    """
    Three-panel enzyme allocation summary:
      (1) f_PEPC vs f_Rubisco,BS
      (2) f_PEPC vs f_Rubisco,Mes
      (3) f_Rubisco,BS vs f_Rubisco,Mes
    Color = Assimilation rate (μmol m⁻² s⁻¹)
    """
    (f_pepc, f_bs, f_mes), fitness, gens = _get_valid_arrays(
        history, ["f_pepc", "f_rubisco_bs", "f_rubisco_mes"]
    )
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=np.nanmin(fitness), vmax=np.nanmax(fitness))

    with _rc_context():
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        pairs = [
            (f_pepc, f_bs, "fₚₑₚc", "fᵣᵤbisco,BS"),
            (f_pepc, f_mes, "fₚₑₚc", "fᵣᵤbisco,Mes"),
            (f_bs, f_mes, "fᵣᵤbisco,BS", "fᵣᵤbisco,Mes"),
        ]
        for (x, y, xl, yl), ax in zip(pairs, axes):
            sc = ax.scatter(
                x, y, c=fitness, cmap=cmap, norm=norm, s=70, edgecolor="k", lw=0.3
            )
            ax.plot(x, y, "-", color="gray", alpha=0.4, lw=1.0)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.grid(True, ls="--", alpha=0.4)
        _add_horizontal_colorbar(
            fig,
            cm.ScalarMappable(norm=norm, cmap=cmap),
            label="Assimilation rate (μmol m⁻² s⁻¹)",
            bottom=0.03,
        )
        fig.suptitle("Enzyme Allocation Strategy Across Evolution", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()


# ==============================================================
#  OPTION 2 — Interactive 3D trajectory (Plotly)
# ==============================================================


def plot_3d_evolution_interactive(history, cmap_name="plasma", save_html=None):
    """
    Interactive 3D visualization of the full enzyme-fraction evolution:
      - X-axis: f_PEPC
      - Y-axis: f_Rubisco,BS
      - Z-axis: f_Rubisco,Mes
      - Color: Total assimilation (A_bs + A_mes)
      - Size:  Total assimilation magnitude
    """
    mean_traits = history.get("mean_traits_per_gen", [])
    fitness = np.asarray(history.get("mean_fitness_per_gen", []), float)

    f_pepc = np.array([t.get("f_pepc", np.nan) for t in mean_traits], float)
    f_bs = np.array([t.get("f_rubisco_bs", np.nan) for t in mean_traits], float)
    f_mes = np.array([t.get("f_rubisco_mes", np.nan) for t in mean_traits], float)

    mask = (
        np.isfinite(f_pepc)
        & np.isfinite(f_bs)
        & np.isfinite(f_mes)
        & np.isfinite(fitness)
    )
    f_pepc, f_bs, f_mes, fitness = f_pepc[mask], f_bs[mask], f_mes[mask], fitness[mask]

    if len(fitness) < 3:
        print("⚠️ Not enough valid points to plot 3D trajectory.")
        return

    # Normalize color scale and marker size
    fitness_min, fitness_max = np.nanmin(fitness), np.nanmax(fitness)
    fitness_norm = (fitness - fitness_min) / (fitness_max - fitness_min + 1e-9)
    size = 6 + 10 * fitness_norm

    # --- Main interactive 3D figure ---
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=f_pepc,
                y=f_bs,
                z=f_mes,
                mode="lines+markers",
                line=dict(color=fitness, colorscale=cmap_name, width=6),
                marker=dict(
                    size=size,
                    color=fitness,
                    colorscale=cmap_name,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text="Assimilation (μmol m⁻² s⁻¹)", font=dict(size=13)
                        ),
                        orientation="h",
                        x=0.5,
                        y=-0.2,  # below the plot
                        xanchor="center",
                        yanchor="top",
                        len=0.7,
                        thickness=18,
                    ),
                ),
                name="Trajectory",
            ),
            go.Scatter3d(
                x=[f_pepc[0]],
                y=[f_bs[0]],
                z=[f_mes[0]],
                mode="markers+text",
                name="Start (C₃-like)",
                marker=dict(color="crimson", size=10, symbol="cross"),
                text=["Start"],
                textposition="top right",
            ),
            go.Scatter3d(
                x=[f_pepc[-1]],
                y=[f_bs[-1]],
                z=[f_mes[-1]],
                mode="markers+text",
                name="End (C₄-like)",
                marker=dict(color="lime", size=10, symbol="diamond-open"),
                text=["End"],
                textposition="bottom left",
            ),
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="fₚₑₚc (PEPC fraction)",
            yaxis_title="fᵣᵤbisco,BS (Bundle Sheath)",
            zaxis_title="fᵣᵤbisco,Mes (Mesophyll)",
            xaxis=dict(nticks=6, backgroundcolor="white", gridcolor="lightgray"),
            yaxis=dict(nticks=6, backgroundcolor="white", gridcolor="lightgray"),
            zaxis=dict(nticks=6, backgroundcolor="white", gridcolor="lightgray"),
        ),
        width=850,
        height=720,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Interactive 3D Enzyme-Fraction Evolution",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
    )

    if save_html:
        fig.write_html(save_html)

    fig.show()


# ==============================================================
#  OPTION 3 — Hybrid layout (3D + 2D projections)
# ==============================================================


def plot_hybrid_evolution_view(history, cmap_name="plasma", save_path=None):
    """Hybrid view: real assimilation + all enzyme fractions."""
    (f_pepc, f_bs, f_mes), fitness, gens = _get_valid_arrays(
        history, ["f_pepc", "f_rubisco_bs", "f_rubisco_mes"]
    )
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=np.nanmin(fitness), vmax=np.nanmax(fitness))

    with _rc_context():
        fig = plt.figure(figsize=(16, 5))
        ax3d = fig.add_subplot(131, projection="3d")

        for i in range(len(f_pepc) - 1):
            ax3d.plot(
                [f_pepc[i], f_pepc[i + 1]],
                [f_bs[i], f_bs[i + 1]],
                [f_mes[i], f_mes[i + 1]],
                color=cmap(norm(fitness[i])),
                lw=2.2,
            )

        ax3d.set_xlabel("fₚₑₚc")
        ax3d.set_ylabel("fᵣᵤbisco,BS")
        ax3d.set_zlabel("fᵣᵤbisco,Mes")

        # 2D projections
        ax_xy = fig.add_subplot(132)
        ax_xy.scatter(
            f_pepc, f_bs, c=fitness, cmap=cmap, norm=norm, s=60, edgecolor="k", lw=0.3
        )
        ax_xy.set_xlabel("fₚₑₚc")
        ax_xy.set_ylabel("fᵣᵤbisco,BS")

        ax_xz = fig.add_subplot(133)
        ax_xz.scatter(
            f_pepc, f_mes, c=fitness, cmap=cmap, norm=norm, s=60, edgecolor="k", lw=0.3
        )
        ax_xz.set_xlabel("fₚₑₚc")
        ax_xz.set_ylabel("fᵣᵤbisco,Mes")

        _add_horizontal_colorbar(
            fig,
            cm.ScalarMappable(norm=norm, cmap=cmap),
            label="Assimilation (μmol m⁻² s⁻¹)",
            bottom=0.02,
        )
        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()
