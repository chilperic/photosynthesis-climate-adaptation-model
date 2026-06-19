#!/usr/bin/env python
"""Run a very short evolutionary optimization smoke test.

The full postdoc analyses used longer simulations. This script uses tiny
settings so that users can verify the workflow quickly.
"""

from photosynthesis_adaptation.parameters import load_parameters
from photosynthesis_adaptation.utils.evolution import run_evolution_simulation


def main() -> None:
    params = load_parameters(environment="default")
    traits = ["alpha_mes", "alpha_bs", "k_l"]
    bounds = [(0.05, 0.95), (0.05, 0.95), (1e-9, 1e-5)]

    result = run_evolution_simulation(
        evolving_traits=traits,
        bounds=bounds,
        base_params=params,
        n_generations=2,
        pop_size=4,
        sigma0=0.2,
        parallel_timeout=30,
        verbose=False,
    )
    print("returned_type:", type(result).__name__)
    if isinstance(result, dict):
        print("keys:", sorted(result.keys())[:10])


if __name__ == "__main__":
    main()
