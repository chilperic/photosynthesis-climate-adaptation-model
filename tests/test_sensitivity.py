import numpy as np
import pytest

from photosynthesis_adaptation.utils.sensitivity_analysis import (
    compute_local_sensitivity,
    compute_sobol_sensitivity,
)


class LinearModel:
    def __init__(self, params):
        self.params = params

    def build(self):
        pass

    def run(self):
        return dict(
            success=True,
            co2_assimilation_rate_bs=self.params["x"] + 2 * self.params["y"],
            co2_assimilation_rate_mes=0.0,
        )


class FailedModel(LinearModel):
    def run(self):
        raise RuntimeError("deliberate failed solve")


def test_sobol_known_additive_indices_and_reproducible_design():
    args = (["x", "y"], [(0, 1), (0, 1)], {"x": 0.5, "y": 0.5})
    a = compute_sobol_sensitivity(*args, n_samples=512, seed=9, model_factory=LinearModel)
    b = compute_sobol_sensitivity(*args, n_samples=512, seed=9, model_factory=LinearModel)
    np.testing.assert_allclose(a["S1"], [0.2, 0.8], atol=0.03)
    np.testing.assert_allclose(a["ST"], [0.2, 0.8], atol=0.03)
    np.testing.assert_array_equal(a["sample_values"], b["sample_values"])
    np.testing.assert_array_equal(a["S1_conf"], b["S1_conf"])
    np.testing.assert_allclose(
        a["outputs"], a["sample_values"][:, 0] + 2 * a["sample_values"][:, 1]
    )


def test_failed_design_is_never_imputed():
    with pytest.raises(RuntimeError, match="No indices computed"):
        compute_sobol_sensitivity(
            ["x", "y"], [(0, 1)] * 2, {"x": 0.5, "y": 0.5}, n_samples=4, model_factory=FailedModel
        )


def test_local_derivatives_at_upper_bound():
    assert compute_local_sensitivity(
        ["x", "y"], [(0, 1)] * 2, {"x": 1.0, "y": 1.0}, 0.01, model_factory=LinearModel
    ) == pytest.approx({"x": 1.0, "y": 2.0})


def test_invalid_design_or_overwritten_traits_rejected():
    with pytest.raises(ValueError, match="power of two"):
        compute_sobol_sensitivity(["x"], [(0, 1)], {"x": 0.5}, n_samples=1000)
    with pytest.raises(ValueError, match="overwritten"):
        compute_sobol_sensitivity(["v_pmax"], [(0, 1)], {"v_pmax": 0.5})
