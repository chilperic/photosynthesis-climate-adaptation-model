"""Backward-compatible parameter-loader wrapper.

New code should use :mod:`photosynthesis_adaptation.parameters`.
"""

from __future__ import annotations

from photosynthesis_adaptation.parameters import (
    available_parameter_sets,
    describe_available_presets,
    load_parameters as _load_parameters,
)


def load_parameters(
    biochemical_params="default",
    environmental_params="default",
    temp_dependence_params="default",
    soil_params="loamy_sand",
    hydraulic_params="default",
    heat_params="default",
):
    return _load_parameters(
        biochemical=biochemical_params,
        environment=environmental_params,
        temperature=temp_dependence_params,
        soil=soil_params,
        hydraulic=hydraulic_params,
        heat=heat_params,
    )
