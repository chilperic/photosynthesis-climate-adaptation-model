"""Parameter loading utilities.

The original postdoc code read JSON files from a working-directory-relative
``resources/`` folder. This module makes the same parameter presets available
from package resources, so scripts can be executed from any directory.
"""

from __future__ import annotations

from importlib import resources
import json
from typing import Any, Dict


_RESOURCE_PACKAGE = "photosynthesis_adaptation.resources"


def _read_json(filename: str) -> Any:
    with resources.files(_RESOURCE_PACKAGE).joinpath(filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _entry_name(entry: Dict[str, Any]) -> str | None:
    """Return the preset name for parameter files using either param_set or soil_type."""
    return entry.get("param_set") or entry.get("soil_type")


def _select_param_set(filename: str, param_set: str) -> Dict[str, float]:
    entries = _read_json(filename)
    if isinstance(entries, dict):
        if param_set in entries:
            return dict(entries[param_set])
        raise ValueError(f"Parameter set '{param_set}' not found in {filename}.")

    for entry in entries:
        if _entry_name(entry) == param_set:
            return {
                key: value
                for key, value in entry.items()
                if key not in {"param_set", "soil_type"}
            }

    available = [_entry_name(entry) for entry in entries if isinstance(entry, dict)]
    raise ValueError(
        f"Parameter set '{param_set}' not found in {filename}. "
        f"Available sets: {available}"
    )


def available_parameter_sets(filename: str) -> list[str]:
    """Return available preset names for a package JSON resource."""
    entries = _read_json(filename)
    if isinstance(entries, dict):
        return sorted(entries)
    return [
        name
        for entry in entries
        if isinstance(entry, dict) and (name := _entry_name(entry)) is not None
    ]


def load_parameters(
    biochemical: str = "default",
    environment: str = "default",
    temperature: str = "default",
    soil: str = "loamy_sand",
    hydraulic: str = "default",
    heat: str = "default",
) -> Dict[str, float]:
    """Load and merge biochemical, environmental, hydraulic, heat, and soil parameters."""
    params: Dict[str, float] = {}
    params.update(_select_param_set("biochemical_parameters.json", biochemical))
    params.update(_select_param_set("environmental_parameters.json", environment))
    params.update(_select_param_set("temp_dependence_parameters.json", temperature))
    params.update(_select_param_set("soil_parameters.json", soil))
    params.update(_select_param_set("hydraulic_parameters.json", hydraulic))
    params.update(_select_param_set("heat_parameters.json", heat))

    if "v_cmax_mes" in params and "r_mes" not in params:
        params["r_mes"] = 0.015 * params["v_cmax_mes"]
    if "v_cmax_bs" in params and "r_bs" not in params:
        params["r_bs"] = 0.015 * params["v_cmax_bs"]
    if "r_mes" in params and "r_bs" in params:
        params["r_d"] = params["r_mes"] + params["r_bs"]

    return params


def describe_available_presets() -> dict[str, list[str]]:
    """Return available presets grouped by parameter file."""
    files = [
        "biochemical_parameters.json",
        "environmental_parameters.json",
        "temp_dependence_parameters.json",
        "soil_parameters.json",
        "hydraulic_parameters.json",
        "heat_parameters.json",
    ]
    return {filename: available_parameter_sets(filename) for filename in files}
