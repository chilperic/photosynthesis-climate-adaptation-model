import casadi as ca


class HeatModel:
    def __init__(self, opti, symbolic_params, variables, hydraulic_output):
        self.opti = opti
        self.symbolic_params = symbolic_params
        self.variables = variables
        self.hydraulic_output = hydraulic_output
        self.output = None

    def build_model(self):
        """Build the heat model for the calculation of compartment temperatures."""

        # ----- Relevant Variables -----
        temp_mes = self.variables["temp_mes"]  # K (Temperature of mesophyll)
        temp_bs = self.variables["temp_bs"]  # K (Temperature of bundle sheath)

        # ----- Relevant Parameters -----
        temp_air = self.symbolic_params["temp"]  # K (Ambient air temperature)
        solar_irradiance = self.symbolic_params[
            "solar_irradiance"
        ]  # W/m^2 (Solar irradiance)

        f_canopy = self.symbolic_params["f_canopy"]  # dimensionless (canopy fraction)
        alpha_mes = self.symbolic_params[
            "alpha_mes"
        ]  # dimensionless (mesophyll absorptance)
        tau_mes = self.symbolic_params[
            "tau_mes"
        ]  # dimensionless (mesophyll transmittance)
        gamma_mes = self.symbolic_params[
            "gamma_mes"
        ]  # dimensionless (mesophyll reabsorption)
        alpha_bs = self.symbolic_params[
            "alpha_bs"
        ]  # dimensionless (bundle sheath absorptance)
        tau_bs = self.symbolic_params[
            "tau_bs"
        ]  # dimensionless (bundle sheath transmittance)
        gamma_bs = self.symbolic_params[
            "gamma_bs"
        ]  # dimensionless (bundle sheath reabsorption)
        rho_soil = self.symbolic_params["rho_soil"]  # dimensionless (soil reflectance)

        u_a = self.symbolic_params[
            "u_a"
        ]  # W/m^2/K (conductive heat transfer coefficient between
        # the mesophyll and the bundle sheath)
        g_h = self.symbolic_params["g_h"]  # mol/m^2/s (boundary layer conductance)
        c_p = self.symbolic_params["c_p"]  # J/(mol*K) (molar heat capacity of air)
        lambda_heat = self.symbolic_params[
            "lambda_heat"
        ]  # J/mol (latent heat of vaporization of
        # water)
        leaf_emissivity = self.symbolic_params["leaf_emissivity"]
        sigma = 5.67e-8  # W/m^2/K^4 (Boltzmann constant)

        transpiration_rate = self.hydraulic_output["transpiration_rate"]  # mol/m^2/s

        # ----- Compartment Irradiance -----
        irradiance_mes, irradiance_bs = _calculate_compartment_irradiance(
            solar_irradiance,
            f_canopy,
            alpha_mes,
            tau_mes,
            alpha_bs,
            tau_bs,
            rho_soil,
            gamma_mes,
            gamma_bs,
        )

        # ----- Cooling Mechanisms -----
        convective_cooling = c_p * g_h * (temp_mes - temp_air)
        transpirative_cooling = lambda_heat * transpiration_rate
        conductive_heat_transfer = u_a * (temp_mes - temp_bs)

        radiative_cooling_mes = leaf_emissivity * sigma * (temp_mes**4 - temp_air**4)
        radiative_cooling_bs = leaf_emissivity * sigma * (temp_bs**4 - temp_air**4)

        # ----- Energy Balance -----
        # Energy balance equation (mesophyll temperature)
        self.opti.subject_to(
            irradiance_mes
            - convective_cooling
            - transpirative_cooling
            - conductive_heat_transfer
            - radiative_cooling_mes
            == 0
        )

        # Energy balance equation (bundle sheath temperature)
        self.opti.subject_to(
            irradiance_bs + conductive_heat_transfer - radiative_cooling_bs == 0
        )

        # ----- Output -----
        self.output = {
            "irradiance_mes": irradiance_mes,
            "irradiance_bs": irradiance_bs,
            "convective_cooling": convective_cooling,
            "transpirative_cooling": transpirative_cooling,
            "conductive_heat_transfer": conductive_heat_transfer,
            "radiative_cooling_mes": radiative_cooling_mes,
            "radiative_cooling_bs": radiative_cooling_bs,
        }

    def get_output(self):
        """Return the output of the heat model."""
        return self.output


def _calculate_compartment_irradiance(
    solar_irradiance,
    f_canopy,
    alpha_mes,
    tau_mes,
    alpha_bs,
    tau_bs,
    rho_soil,
    gamma_mes,
    gamma_bs,
):
    """
    Calculate the irradiance absorbed by the mesophyll and bundle sheath compartments.

    Args:
        eta: Fraction of irradiance absorbed by mesophyll (dimensionless)
        solar_irradiance: Total solar irradiance (W/m^2)
    Returns:
        irradiance_mes: Irradiance absorbed by mesophyll (W/m^2)
        irradiance_bs: Irradiance absorbed by bundle sheath (W/m^2)
    """

    soil_reflected_irradiance = _calculate_soil_reflected_irradiance(
        solar_irradiance, f_canopy, rho_soil, tau_mes, tau_bs
    )

    irradiance_mes = _calculate_mesophyll_irradiance(
        solar_irradiance, f_canopy, alpha_mes, gamma_mes, soil_reflected_irradiance
    )

    irradiance_bs = _calculate_bundle_sheath_irradiance(
        solar_irradiance,
        f_canopy,
        alpha_bs,
        gamma_bs,
        tau_mes,
        soil_reflected_irradiance,
    )

    return irradiance_mes, irradiance_bs


def _calculate_soil_reflected_irradiance(
    solar_irradiance, f_canopy, rho_soil, tau_mes, tau_bs
):
    """
    Calculate the irradiance reflected by the soil.

    Args:
        solar_irradiance: Total solar irradiance (W/m^2)
        f_canopy: Fraction of canopy cover (dimensionless)
        rho_soil: Reflectance of soil (dimensionless)
        tau_mes: Transmittance of mesophyll (dimensionless)
        tau_bs: Transmittance of bundle sheath (dimensionless)

    Returns:
        irradiance_refl_soil: Irradiance reflected by soil (W/m^2)
    """

    irradiance_soil = (
        1 - f_canopy
    ) * solar_irradiance + f_canopy * tau_mes * tau_bs * solar_irradiance

    irradiance_refl_soil = rho_soil * irradiance_soil

    return irradiance_refl_soil


def _calculate_mesophyll_irradiance(
    solar_irradiance, f_canopy, alpha_mes, gamma_mes, soil_refelcted_irradiance
):
    """
    Calculate the irradiance absorbed by the mesophyll compartment.

    Args:
        solar_irradiance: Total solar irradiance (W/m^2)
        f_canopy: Fraction of canopy cover (dimensionless)
        alpha_mes: Absorption coefficient of mesophyll (dimensionless)
        gamma_mes: Mesophyll reabsorption (dimensionless)
        soil_reflected_irradiance: Irradiance reflected by soil (W/m^2)

    Returns:
        irradiance_mes: Irradiance absorbed by mesophyll (W/m^2)
    """

    direct_absorption_mes = f_canopy * alpha_mes * solar_irradiance
    reflected_absorption = gamma_mes * soil_refelcted_irradiance

    irradiance_mes = direct_absorption_mes + reflected_absorption

    return irradiance_mes


def _calculate_bundle_sheath_irradiance(
    solar_irradiance, f_canopy, alpha_bs, gamma_bs, tau_mes, soil_reflected_irradiance
):
    """
    Calculate the irradiance absorbed by the bundle sheath compartment.

    Args:
        solar_irradiance: Total solar irradiance (W/m^2)
        f_canopy: Fraction of canopy cover (dimensionless)
        alpha_bs: Absorption coefficient of bundle sheath (dimensionless)
        gamma_bs: Bundle sheath reabsorption (dimensionless)
        tau_mes: Transmittance of mesophyll (dimensionless)
        soil_reflected_irradiance: Irradiance reflected by soil (W/m^2)

    Returns:
        irradiance_bs: Irradiance absorbed by bundle sheath (W/m^2)
    """

    direct_absorption_bs = f_canopy * tau_mes * alpha_bs * solar_irradiance
    reflected_absorption = gamma_bs * soil_reflected_irradiance

    irradiance_bs = direct_absorption_bs + reflected_absorption

    return irradiance_bs
