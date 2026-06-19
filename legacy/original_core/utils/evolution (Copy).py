# The following code was fully provided by Dr. Chilperic Armel Foko Kuate and only slightly modified
# afterwards.


from cma import CMAEvolutionStrategy
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import logging
from model.model_builder import ModelBuilder
import numpy as np
from tqdm import tqdm

# Configure logger for warnings and errors
logging.basicConfig(
    filename="solver_errors.log",
    filemode="a",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def run_evolution_simulation(evolving_traits, bounds, base_params, n_generations=100, pop_size=16,
                             sigma0=0.3):
    '''
    Run CMA-ES optimization over specified generations and traits.

    Args:
        evolving_traits (list): List of trait names to evolve.
        bounds (list): List of bounds for each trait.
        base_params (dict): Base parameters for the model.
        n_generations (int, optional): Number of generations to run. Defaults to 100.
        pop_size (int, optional): Population size for CMA-ES. Defaults to 16.
        sigma0 (float, optional): Initial standard deviation for CMA-ES. Defaults to 0.3.
    '''

    # Initialize
    num_traits = len(evolving_traits)
    x0 = np.random.rand(num_traits)
    es = CMAEvolutionStrategy(
        x0.tolist(),
        sigma0,
        {
            "bounds": [0.0, 1.0],
            "popsize": pop_size,
            "maxiter": n_generations,
            "verb_log": 0,
            "verb_disp": 0
        }
    )

    # Data containers
    trait_history, delta13C_history = [], []
    delta13C_gen_list = []  # New list to store the generation for each delta13C
    fitness_bs_history, fitness_mes_history = [], []

    total_evals = 0
    accepted_evals = 0

    with tqdm(total=n_generations, desc="Evolving", leave=True) as pbar:
        current_gen = 0
        while not es.stop():
            solutions = es.ask()
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(
                    lambda x: _evaluate_individual(x, evolving_traits, base_params, bounds),
                    solutions
                ))

            # Unpack results
            fitnesses, delta13Cs, f_bs_list, f_mes_list, successes = zip(*results)

            # Record data per solution
            for x, traits, f, d13c, fb, fm, success in zip(
                solutions,
                [dict(zip(evolving_traits, _denormalize_vector(x, bounds))) for x in solutions],
                fitnesses,
                delta13Cs,
                f_bs_list,
                f_mes_list,
                successes
            ):
                trait_history.append(traits)
                delta13C_history.append(d13c)
                delta13C_gen_list.append(current_gen)  # Record current generation
                fitness_bs_history.append(fb)
                fitness_mes_history.append(fm)
                total_evals += 1
                if success:
                    accepted_evals += 1

            # Prepare fitness for CMA-ES (minimize)
            fitness_for_es = [-f if np.isfinite(f) else 1e6 for f in fitnesses]
            es.tell(solutions, fitness_for_es)

            current_gen += 1
            pbar.update(1)

        print(
            f"\nTotal evaluations: {total_evals}, "
            f"Accepted: {accepted_evals}, "
            f"Rejected: {total_evals - accepted_evals}"
        )

    # Aggregate final fitness
    fitness_combined = [
        _validate_fitness_value(fb + fm) for fb, fm in zip(fitness_bs_history, fitness_mes_history)
    ]

    return {
        "traits": trait_history,
        "fitness": fitness_combined,
        "delta13C": delta13C_history,
        "delta13C_generations": delta13C_gen_list,  # Return the per-individual generation list
        "fitness_bs": fitness_bs_history,
        "fitness_mes": fitness_mes_history,
        "generations": list(range(n_generations)),  # Can be kept for reference if needed
        "final_fitness": [f for f in fitness_combined[-10:] if np.isfinite(f)],
    }


def _evaluate_individual(param_vector, evolving_traits, base_params, bounds):
    '''
    Evaluate a candidate solution, returning fitness and δ13C.

    Args:
        param_vector (list): Normalized trait values.
        evolving_traits (list): List of trait names.
        base_params (dict): Base parameters for the model.
        bounds (list): Bounds for each trait.

    Returns:
        tuple: Total fitness, δ13C value, fitness for BS, fitness for MES, success flag.
    '''

    traits = dict(zip(evolving_traits, _denormalize_vector(param_vector, bounds)))
    params = deepcopy(base_params)
    params.update(traits)
    try:
        model = ModelBuilder(params)
        model.build()
        result = model.run()
        f_co2_bs = result["co2_assimilation_rate_bs"]
        f_co2_mes = result["co2_assimilation_rate_mes"]
        total_fitness = _validate_fitness_value(f_co2_bs + f_co2_mes)
        delta13C = -27 + 14 * (f_co2_bs / total_fitness) if total_fitness > 0 else np.nan
        return total_fitness, delta13C, f_co2_bs, f_co2_mes, True
    except Exception as e:
        logging.warning(f"Evaluation failed: {e}")
        return -1e6, np.nan, np.nan, np.nan, False
    

def _denormalize_vector(p, bounds):
    '''
    Convert normalized parameters back to original bounds.

    Args: 
        p (list): Normalized parameter values.
        bounds (list): List of bounds for each trait.

    Returns:
        list: Denormalized parameter values.
    '''

    return [
        p[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0] for i in range(len(p))
    ]

def _validate_fitness_value(f):
    '''
    Ensure fitness is within acceptable bounds.

    Args:
        f (float): Fitness value to validate.

    Returns:
        float: Validated fitness value or NaN.
    '''

    return f if np.isfinite(f) and 0 <= f <= 100 else np.nan