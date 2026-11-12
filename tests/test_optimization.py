import numpy as np
import pytest

from photosynthesis_adaptation.model.model_builder import ModelBuilder
from photosynthesis_adaptation.parameters import load_parameters
from photosynthesis_adaptation.utils.evolution import optimize_traits


def objective(x, names, params, verbose):
    fitness = -float(np.sum((x - 0.3) ** 2))
    return fitness, np.nan, fitness, 0.0, True


def test_seeded_optimization_records_actual_evaluated_traits():
    kw = dict(
        evolving_traits=["a", "b"],
        bounds=[(0, 1), (0, 1)],
        base_params={},
        n_generations=5,
        pop_size=6,
        seed=7,
        evaluator=objective,
    )
    a, b = optimize_traits(**kw), optimize_traits(**kw)
    assert a["traits"] == b["traits"]
    assert a["fitness"] == b["fitness"]
    for traits, fitness in zip(a["traits"], a["fitness"]):
        assert fitness == pytest.approx(-sum((v - 0.3) ** 2 for v in traits.values()))


def test_failures_do_not_become_observations():
    def sometimes(x, *args):
        return (99.0, 0.0, 0.0, 0.0, False) if x[0] < 0.5 else objective(x, *args)

    out = optimize_traits(
        ["a", "b"], [(0, 1)] * 2, {}, n_generations=1, pop_size=20, evaluator=sometimes, seed=4
    )
    assert not all(out["success"])
    for fit, ok in zip(out["fitness"], out["success"]):
        assert np.isfinite(fit) == ok
    assert out["mean_fitness_per_gen"][0] == pytest.approx(np.nanmean(out["fitness"]))
    with pytest.raises(RuntimeError, match="No successful"):
        optimize_traits(["a", "b"], [(0, 1)] * 2, {}, evaluator=lambda *a: (0, 0, 0, 0, False))


def test_overwritten_traits_are_rejected():
    with pytest.raises(ValueError, match="overwritten"):
        optimize_traits(["v_pmax"], [(1, 10)], load_parameters())


def test_failure_is_explicit_and_input_params_are_copied():
    p = load_parameters()
    model = ModelBuilder(p)
    p["co2_atm"] = -999
    assert model.params["co2_atm"] != -999
    with pytest.raises(RuntimeError, match="build"):
        model.run()
    model.build()
    failed = model.run({"max_iter": 0}, raise_on_failure=False)
    assert failed["success"] is False
    assert np.isnan(failed["assimilation_total"])


def test_real_default_ipopt_solve():
    model = ModelBuilder(load_parameters())
    model.build()
    out = model.run()
    assert out["success"]
    assert np.isfinite(out["assimilation_total"])
    assert out["max_constraint_violation"] < 1e-4
    assert out["assimilation_total"] == pytest.approx(
        1e6 * (out["co2_assimilation_rate_bs"] + out["co2_assimilation_rate_mes"])
    )
