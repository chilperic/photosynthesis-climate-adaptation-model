# ==============================================================
#  Robust Hybrid CMA–Kimura Evolution (Three-Enzyme-Fraction Aware)
#  Author: Dr. Chilperic Armel Foko Kuate — 2025-10-29
# ==============================================================

import concurrent.futures
from cma import CMAEvolutionStrategy
from concurrent.futures import TimeoutError
from copy import deepcopy
import logging
import numpy as np
from tqdm import tqdm
import warnings
from model.model_builder import ModelBuilder

# --------------------------------------------------------------
# Configuration and Logging
# --------------------------------------------------------------
warnings.filterwarnings("ignore", message=r"Sampling standard deviation", module="cma")

logging.basicConfig(
    filename="solver_errors.log",
    filemode="a",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ==============================================================
#  Main Evolution Function
# ==============================================================


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
):
    """
    Hybrid CMA-Kimura evolution maximizing total CO₂ assimilation (A_bs + A_mes).
    Compatible with enzyme-fraction-aware ModelBuilder (f_pepc, f_rubisco_mes, f_rubisco_bs).
    """

    num_traits = len(evolving_traits)
    bounds = np.maximum(np.array(bounds, float), 1e-8)
    log_mode = [
        np.all(np.array(b) > 1.1) for b in bounds
    ]  # log-transform for wide ranges

    # ----------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------
    if initial_traits is not None:
        init_vec = np.array([initial_traits[t] for t in evolving_traits])
        x0 = _normalize_vector(init_vec, bounds, log_mode)
    else:
        x0 = np.random.uniform(0.3, 0.7, num_traits)

    population = np.clip(
        np.random.normal(loc=x0, scale=0.05, size=(pop_size, num_traits)),
        1e-6,
        1 - 1e-6,
    )

    es = CMAEvolutionStrategy(
        x0.tolist(),
        sigma0,
        {
            "bounds": [0.0, 1.0],
            "popsize": pop_size,
            "tolfun": 1e-8,
            "CSA_dampfac": 2.5,
            "verb_log": 0,
            "verb_disp": 0,
        },
    )

    # ----------------------------------------------------------
    # Storage
    # ----------------------------------------------------------
    trait_history, delta13C_history, delta13C_gen_list = [], [], []
    fitness_bs_history, fitness_mes_history, fitness_total_history = [], [], []
    mean_fitness_per_gen, mean_traits_per_gen = [], []
    sigma_trace, cov_trace = [], []
    cached_fitness = {}

    # ==============================================================
    #  Parallel evaluation with fallback
    # ==============================================================
    def _evaluate_population(pop):
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(4, pop_size)) as ex:
            futures = []
            for x in pop:
                key = tuple(np.round(x, 4))
                if key in cached_fitness:
                    results.append(cached_fitness[key])
                    continue
                fut = ex.submit(
                    _evaluate_individual,
                    _denormalize_vector(x, bounds, log_mode),
                    evolving_traits,
                    base_params,
                    verbose,
                )
                futures.append((fut, key))

            for fut, key in futures:
                try:
                    res = fut.result(timeout=parallel_timeout)
                except TimeoutError:
                    logging.warning("Evaluation timeout — skipping.")
                    res = (-1e3, np.nan, np.nan, np.nan, False)
                except Exception as e:
                    logging.warning(f"Evaluation error: {e}")
                    res = (-1e3, np.nan, np.nan, np.nan, False)
                cached_fitness[key] = res
                results.append(res)

        # Fallback if no valid results
        if len(results) < pop_size or not np.any(np.isfinite([r[0] for r in results])):
            logging.warning("Fallback serial evaluation triggered.")
            for x in pop:
                key = tuple(np.round(x, 4))
                if key in cached_fitness:
                    results.append(cached_fitness[key])
                    continue
                try:
                    res = _evaluate_individual(
                        _denormalize_vector(x, bounds, log_mode),
                        evolving_traits,
                        base_params,
                        verbose,
                    )
                except Exception as e:
                    logging.warning(f"Serial eval failed: {e}")
                    res = (-1e3, np.nan, np.nan, np.nan, False)
                cached_fitness[key] = res
                results.append(res)
        return results

    # ==============================================================
    #  Evolution loop
    # ==============================================================
    with tqdm(total=n_generations, desc="Hybrid evolution", leave=True) as pbar:
        for gen in range(n_generations):
            results = _evaluate_population(population)
            fitnesses, delta13Cs, f_bs_list, f_mes_list, successes = zip(*results)
            fitnesses = np.array(fitnesses, float)

            finite_mask = np.isfinite(fitnesses) & (fitnesses > -1e5)
            if not np.any(finite_mask):
                logging.warning(
                    f"No valid individuals at generation {gen}. Re-sampling."
                )
                population = np.random.rand(pop_size, num_traits)
                continue

            f_valid = fitnesses[finite_mask]
            probs = np.exp(beta * (f_valid - np.max(f_valid)))
            probs /= np.sum(probs)

            n_elite = max(2, int(elite_frac * np.sum(finite_mask)))
            elite_population = population[finite_mask][np.argsort(f_valid)[-n_elite:]]

            es.mean = np.mean(elite_population, axis=0)
            es.C = np.cov(elite_population.T) + 1e-5 * np.eye(num_traits)
            es.sigma = float(np.clip(es.sigma * 0.95, 0.05, 0.8))
            cov_trace.append(np.trace(es.C))
            sigma_trace.append(es.sigma)

            # Offspring generation
            new_population = []
            for _ in range(pop_size):
                offspring = np.random.multivariate_normal(es.mean, es.sigma**2 * es.C)
                offspring = np.clip(offspring, 1e-6, 1 - 1e-6)
                new_population.append(offspring)
            population = np.array(new_population)

            # ----------------------------------------------------------
            # Recording
            # ----------------------------------------------------------
            for x, f, d13c, fb, fm in zip(
                population, fitnesses, delta13Cs, f_bs_list, f_mes_list
            ):
                trait_dict = dict(
                    zip(evolving_traits, _denormalize_vector(x, bounds, log_mode))
                )
                trait_history.append(trait_dict)
                delta13C_history.append(d13c)
                delta13C_gen_list.append(gen)
                fitness_total_history.append(f)
                fitness_bs_history.append(fb)
                fitness_mes_history.append(fm)

            mean_fitness_per_gen.append(np.nanmean(fitnesses))
            mean_trait_values = np.nanmean(
                [_denormalize_vector(x, bounds, log_mode) for x in population], axis=0
            )
            mean_traits_per_gen.append(dict(zip(evolving_traits, mean_trait_values)))

            best_fit = np.nanmax(fitnesses)
            pbar.set_postfix(
                {
                    "mean_fit": f"{np.nanmean(fitnesses):.2f}",
                    "best_fit": f"{best_fit:.2f}",
                    "σ": f"{es.sigma:.2f}",
                }
            )
            pbar.update(1)

            if verbose:
                print(f"Generation {gen}: best_fit={best_fit:.3f}")

    # ==============================================================
    #  Return results
    # ==============================================================
    return {
        "traits": trait_history,
        "fitness": fitness_total_history,
        "fitness_bs": fitness_bs_history,
        "fitness_mes": fitness_mes_history,
        "delta13C": delta13C_history,
        "delta13C_generations": delta13C_gen_list,
        "mean_fitness_per_gen": mean_fitness_per_gen,
        "mean_traits_per_gen": mean_traits_per_gen,
        "sigma_trace": sigma_trace,
        "cov_trace": cov_trace,
        "generations": list(range(len(mean_fitness_per_gen))),
    }


# ==============================================================
#  Helper Functions
# ==============================================================


def _evaluate_individual(param_vector, evolving_traits, base_params, verbose=False):
    """Evaluate a single phenotype with three-fraction enzyme budget."""
    traits = dict(zip(evolving_traits, param_vector))
    params = deepcopy(base_params)

    # Skip invalids quickly
    if any((not np.isfinite(v)) or (v <= 0) for v in traits.values()):
        return -1e3, np.nan, np.nan, np.nan, False

    params.update(traits)

    # Normalize enzyme fractions if present
    if all(f in params for f in ["f_pepc", "f_rubisco_mes", "f_rubisco_bs"]):
        total_frac = params["f_pepc"] + params["f_rubisco_mes"] + params["f_rubisco_bs"]
        if total_frac > 1e-9:
            scale = 1.0 / total_frac
            params["f_pepc"] *= scale
            params["f_rubisco_mes"] *= scale
            params["f_rubisco_bs"] *= scale

    try:
        model = ModelBuilder(params)
        model.build()
        result = model.run()

        f_bs = float(result.get("co2_assimilation_rate_bs", 0.0))
        f_mes = float(result.get("co2_assimilation_rate_mes", 0.0))
        total_f = f_bs + f_mes

        # Check for valid assimilation
        if not np.isfinite(total_f) or total_f <= 1e-9:
            return -1e3, np.nan, f_bs, f_mes, False

        # Robust δ13C proxy (C3 ≈ -27‰ → C4 ≈ -13‰)
        frac_bs = f_bs / (total_f + 1e-9)
        delta13C = np.clip(-27 + 14 * frac_bs, -40, -5)

        if verbose:
            print(
                f"[Eval OK] A_bs={f_bs:.3e}, A_mes={f_mes:.3e}, total={total_f:.3e}, δ13C={delta13C:.2f}"
            )

        return total_f, delta13C, f_bs, f_mes, True

    except Exception as e:
        logging.warning(f"Model evaluation failed: {e}")
        return -1e3, np.nan, np.nan, np.nan, False


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
