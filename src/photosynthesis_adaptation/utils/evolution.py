"""Seeded CMA-ES trait optimization, not a population-genetic evolution model.

Fitness values retain the legacy mol m^-2 s^-1 unit. The isotope field is an
illustrative interpolation proxy, not a mechanistic discrimination model.
"""

import logging
import warnings
from copy import deepcopy

import numpy as np
from cma import CMAEvolutionStrategy

from photosynthesis_adaptation.model.model_builder import ModelBuilder

logger = logging.getLogger(__name__)


def optimize_traits(
    evolving_traits,
    bounds,
    base_params,
    n_generations=80,
    pop_size=16,
    sigma0=0.3,
    initial_traits=None,
    verbose=False,
    *,
    seed=0,
    log_scale=None,
    evaluator=None,
):
    """Maximize assimilation with ordered ask/tell evaluations and exact caching.

    Evaluator signature: ``(values, names, params, verbose)`` ->
    ``(fitness, isotope_proxy, bs, mes, success)``. Evaluation is serial; a
    process-pool timeout cannot safely cancel an in-flight native solver.
    """
    names = list(evolving_traits)
    limits = np.asarray(bounds, dtype=float)
    if not names or len(set(names)) != len(names) or limits.shape != (len(names), 2):
        raise ValueError("Unique traits and one pair of bounds per trait are required")
    if not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("Bounds must be finite and strictly increasing")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < minimum
        for v, minimum in [(n_generations, 1), (pop_size, 4), (seed, 0)]
    ):
        raise ValueError("Require integer generations >= 1, population >= 4 and seed >= 0")
    if not np.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("sigma0 must be finite and positive")
    log_mode = (
        np.zeros(len(names), dtype=bool) if log_scale is None else np.asarray(log_scale, dtype=bool)
    )
    if log_mode.shape != (len(names),) or np.any(limits[log_mode] <= 0):
        raise ValueError("log_scale must match traits and requires positive bounds")
    if evaluator is None:
        overwritten = {
            "f_pepc",
            "f_rubisco_mes",
            "f_rubisco_bs",
            "v_pmax",
            "v_cmax_mes",
            "v_cmax_bs",
        }
        if set(names) & overwritten:
            raise ValueError(
                "These traits are internal optimization variables/capacities and would be overwritten"
            )
        if set(names) - set(base_params):
            raise ValueError("Every evolving trait must be an existing model parameter")
        evaluator = _evaluate_individual
    initial = (
        np.mean(limits, axis=1)
        if initial_traits is None
        else np.array([initial_traits[t] for t in names])
    )
    if (
        not np.all(np.isfinite(initial))
        or np.any(initial < limits[:, 0])
        or np.any(initial > limits[:, 1])
    ):
        raise ValueError("initial_traits must lie within bounds")
    rng = np.random.default_rng(seed)
    es = CMAEvolutionStrategy(
        _normalize_vector(initial, limits, log_mode),
        sigma0,
        {
            "bounds": [0.0, 1.0],
            "popsize": pop_size,
            "seed": np.nan,
            "randn": lambda *shape: rng.standard_normal(shape),
            "verbose": -9,
            "verb_log": 0,
            "tolfun": 1e-12,
        },
    )
    out = {
        key: []
        for key in (
            "traits",
            "fitness",
            "fitness_bs",
            "fitness_mes",
            "delta13C",
            "delta13C_generations",
            "mean_fitness_per_gen",
            "mean_traits_per_gen",
            "sigma_trace",
            "cov_trace",
            "generations",
            "success",
        )
    }
    out.update(
        seed=int(seed),
        algorithm="CMA-ES",
        fitness_unit="mol m^-2 s^-1",
        isotope_model="illustrative interpolation proxy",
        evaluations=0,
    )
    cache = {}
    for gen in range(n_generations):
        population = es.ask()
        values = [_denormalize_vector(x, limits, log_mode) for x in population]
        results = []
        for value in values:
            key = tuple(value)
            if key not in cache:
                cache[key] = evaluator(value, names, deepcopy(base_params), verbose)
                out["evaluations"] += 1
            results.append(cache[key])
        valid = np.array([bool(r[4]) and np.isfinite(r[0]) for r in results])
        fitness = np.array([float(r[0]) if ok else np.nan for r, ok in zip(results, valid)])
        if not valid.any():
            raise RuntimeError(f"No successful evaluations in generation {gen}")
        costs = -fitness
        worst = float(np.max(costs[valid]))
        costs[~valid] = worst + max(1.0, abs(worst))
        es.tell(population, costs.tolist())
        for value, result, ok, fit in zip(values, results, valid, fitness):
            out["traits"].append(dict(zip(names, map(float, value))))
            for key, item in zip(
                ("fitness", "delta13C", "fitness_bs", "fitness_mes"),
                (fit, result[1], result[2], result[3]),
            ):
                out[key].append(float(item) if ok else np.nan)
            out["success"].append(bool(ok))
            out["delta13C_generations"].append(gen)
        out["mean_fitness_per_gen"].append(float(np.mean(fitness[valid])))
        out["mean_traits_per_gen"].append(
            dict(zip(names, map(float, np.mean(np.array(values)[valid], axis=0))))
        )
        out["sigma_trace"].append(float(es.sigma))
        out["cov_trace"].append(float(np.trace(es.C)))
        out["generations"].append(gen)
        if es.stop():
            break
    return out


def run_evolution_simulation(
    evolving_traits,
    bounds,
    base_params,
    n_generations=80,
    pop_size=16,
    sigma0=0.3,
    beta=1.0,
    elite_frac=0.25,
    parallel_timeout=60,
    initial_traits=None,
    verbose=False,
    *,
    seed=0,
    log_scale=None,
    evaluator=None,
):
    """Compatibility entry point. Historical trajectories are not reproduced.

    beta, elite_frac and parallel_timeout belonged to the old heuristic; custom
    values are rejected instead of silently ignored. Use optimize_traits.
    """
    if (beta, elite_frac, parallel_timeout) != (1.0, 0.25, 60):
        raise ValueError("Legacy selection/timeout options are unsupported; use optimize_traits")
    warnings.warn(
        "Use optimize_traits: this is CMA-ES optimization, not Kimura evolution",
        DeprecationWarning,
        stacklevel=2,
    )
    return optimize_traits(
        evolving_traits,
        bounds,
        base_params,
        n_generations,
        pop_size,
        sigma0,
        initial_traits,
        verbose,
        seed=seed,
        log_scale=log_scale,
        evaluator=evaluator,
    )


def _evaluate_individual(param_vector, evolving_traits, base_params, verbose=False):
    params = deepcopy(base_params)
    params.update(zip(evolving_traits, map(float, param_vector)))
    try:
        model = ModelBuilder(params)
        model.build()
        result = model.run()
        bs = float(result["co2_assimilation_rate_bs"])
        mes = float(result["co2_assimilation_rate_mes"])
        total = bs + mes
        if not result["success"] or not np.isfinite(total):
            raise RuntimeError("Invalid model fitness")
        proxy = float(np.clip(-27 + 14 * bs / total, -40, -5)) if total > 0 else np.nan
        return total, proxy, bs, mes, True
    except RuntimeError as error:
        logger.warning("Model evaluation failed: %s", error)
        return np.nan, np.nan, np.nan, np.nan, False


def _normalize_vector(vec, bounds, log_mode):
    """Normalize vector to [0,1] using log10 scaling where appropriate."""
    normed = []
    for i, (v, (low, high)) in enumerate(zip(vec, bounds)):
        if log_mode[i]:
            log_low, log_high = np.log10(low), np.log10(high)
            normed.append((np.log10(v) - log_low) / (log_high - log_low))
        else:
            normed.append((v - low) / (high - low))
    return np.clip(normed, 0, 1)


def _denormalize_vector(norm_vec, bounds, log_mode):
    """Inverse of _normalize_vector."""
    denormed = []
    for i, (n, (low, high)) in enumerate(zip(norm_vec, bounds)):
        n = np.clip(n, 0, 1)
        if log_mode[i]:
            log_low, log_high = np.log10(low), np.log10(high)
            denormed.append(10 ** (log_low + n * (log_high - log_low)))
        else:
            denormed.append(low + n * (high - low))
    return np.array(denormed)
