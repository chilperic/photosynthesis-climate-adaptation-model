import casadi as ca


class HydraulicModel:
    def __init__(self, opti, symbolic_params, variables):
        self.opti = opti
        self.symbolic_params = symbolic_params
        self.variables = variables
        self.output = None

    def build_model(self):
        """Build the hydraulic model for the constraint on water availability."""

        # ----- Relevant Variables -----
        g_s = self.variables["g_s"]  # mol/m^2/s (stomatal conductance)

        # ----- Relevant Parameters -----
        rel_soil_wetness = self.symbolic_params[
            "rel_soil_wetness"
        ]  # (relative soil wetness)
        psi_sat = self.symbolic_params["psi_sat"]  # Pa (saturated soil water potential)
        b = self.symbolic_params["b"]  # dimensionless (Campbell's b parameter)
        leaf_water_potential = self.symbolic_params[
            "leaf_water_potential"
        ]  # Pa (leaf water
        # potential)
        k_l = self.symbolic_params["k_l"]  # mol/m^2/s/Pa (leaf hydraulic conductance)
        rel_humidity = self.symbolic_params["rel_humidity"]  # % (relative humidity)
        temp_mes = self.variables["temp_mes"]  # K (temperature measurement)

        # ----- Max Water Uptake -----
        soil_water_potential = _calculate_soil_water_potential(
            rel_soil_wetness, psi_sat, b
        )

        max_water_uptake = _calculate_max_water_uptake(
            soil_water_potential, leaf_water_potential, k_l
        )

        # ----- Transpiration Rate -----
        transpiration_rate = _calculate_transpiration_rate(g_s, rel_humidity, temp_mes)

        # ----- Constraints -----
        self.opti.subject_to(transpiration_rate <= max_water_uptake)

        # ----- Output -----
        self.output = {
            "max_water_uptake": max_water_uptake,
            "transpiration_rate": transpiration_rate,
        }

    def get_output(self):
        """Return the output of the hydraulic model."""
        return self.output


def _calculate_soil_water_potential(rel_soil_wetness, psi_sat, b):
    """
    Calculate the soil water potential (only matric potential) using the Campbell equation.

    Args:
        rel_soil_wetness: Relative soil wetness (0 to 1)
        psi_sat: Saturated soil water potential (Pa)
        b: Campbell's b parameter (unitless)

    Returns:
        soil_water_potential: Soil water potential (Pa)
    """

    return psi_sat * (rel_soil_wetness ** (-b))


def _calculate_max_water_uptake(soil_water_potential, leaf_water_potential, k_l):
    """
    Calculate the maximum water uptake based on soil water potential, leaf water potential, and
    hydraulic conductivity.

    Args:
        soil_water_potential: Soil water potential (Pa)
        leaf_water_potential: Leaf water potential (Pa)
        k_l: Hydraulic conductivity (m/s)

    Returns:
        max_water_uptake: Maximum water uptake (m/s)
    """

    return k_l * (soil_water_potential - leaf_water_potential)


def _calculate_transpiration_rate(g_s, rel_humidity, temp):
    """
    Calculate the transpiration rate in terms of the atmospheric demand from Bonan (2019).

    Args:
        gs: Stomatal conductance (mol/m^2/s)
        temp: Temperature (K)
        rel_humidity: Relative humidity (%)
    Returns:
        transpiration_rate: Transpiration rate (mol/m^2/s)
    """

    barometric_pressure = 101325

    saturation_vapor_pressure = _calculate_saturation_vapor_pressure(temp)
    actual_vapor_pressure = _calculate_actual_vapor_pressure(
        saturation_vapor_pressure, rel_humidity
    )

    transpiration_rate = (
        g_s * (saturation_vapor_pressure - actual_vapor_pressure) / barometric_pressure
    )

    return transpiration_rate


def _calculate_saturation_vapor_pressure(temp):
    """
    Calculate the saturation vapor pressure with Tetens equation from Monteith & Unsworth (2013).

    Args:
        temp: Temperature (K)
    Returns:
        saturation_vapor_pressure: Saturation vapor pressure (Pa)
    """

    return 611 * ca.exp((17.27 * (temp - 273.15)) / (temp - 36))


def _calculate_actual_vapor_pressure(saturation_vapor_pressure, rel_humidity):
    """
    Calculate the actual vapor pressure with equation from Wei et al. (2021).

    Args:
        saturation_vapor_pressure: Saturation vapor pressure (Pa)
        rel_humidity: Relative humidity (%)
    Returns:
        actual_vapor_pressure: Actual vapor pressure (Pa)
    """

    return rel_humidity / 100 * saturation_vapor_pressure
