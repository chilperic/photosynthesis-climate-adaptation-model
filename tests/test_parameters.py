from photosynthesis_adaptation.parameters import load_parameters, describe_available_presets


def test_parameter_presets_are_available():
    presets = describe_available_presets()
    assert "environmental_parameters.json" in presets
    assert "default" in presets["environmental_parameters.json"]
    assert "loamy_sand" in presets["soil_parameters.json"]


def test_load_default_parameters_has_required_keys():
    params = load_parameters()
    required = {
        "solar_irradiance",
        "rel_humidity",
        "rel_soil_wetness",
        "co2_atm",
        "o2_atm",
        "v_cmax_mes",
        "v_cmax_bs",
        "v_pmax",
        "k_l",
        "leaf_water_potential",
    }
    missing = required.difference(params)
    assert not missing
