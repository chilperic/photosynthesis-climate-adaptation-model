import casadi as ca
from photosynthesis_adaptation.utils import unit_conversion as uc


class BiochemicalModel:
    """
    Biochemical submodel using temperature-corrected enzyme kinetics and
    catalytic capacities (Vmax) passed from ModelBuilder.
    """

    def __init__(self, opti, symbolic_params, variables):
        self.opti = opti
        self.symbolic_params = symbolic_params
        self.variables = variables
        self.output = None

    # ===========================================================
    # BUILD MODEL
    # ===========================================================
    def build_model(self):
        """Construct the biochemical model for photosynthetic CO₂ assimilation."""

        # -----------------------------------------------------------
        # Core state variables
        # -----------------------------------------------------------
        co2_mes = self.variables["co2_mes"]  # µmol/m³
        g_s = self.variables["g_s"]  # mol m⁻² s⁻¹
        temp_mes = self.variables["temp_mes"]  # K
        temp_bs = self.variables["temp_bs"]  # K

        # -----------------------------------------------------------
        # Convert gas-phase reference parameters
        # -----------------------------------------------------------
        co2_atm = uc.ubar_to_molar_concentration(self.symbolic_params["co2_atm"])
        o2_atm = uc.ubar_to_molar_concentration(self.symbolic_params["o2_atm"])
        o2_mes = o2_atm

        self.symbolic_params["co2_atm_uc"] = co2_atm
        self.symbolic_params["o2_atm_uc"] = o2_atm
        self.symbolic_params["o2_mes_uc"] = o2_mes

        # -----------------------------------------------------------
        # Enzyme fractions from ModelBuilder (f_pepc, f_rubisco_mes, f_rubisco_bs)
        # -----------------------------------------------------------
        f_pepc = self.variables["f_pepc"]
        f_rubisco_mes = self.variables["f_rubisco_mes"]
        f_rubisco_bs = self.variables["f_rubisco_bs"]

        # -----------------------------------------------------------
        # Enzyme capacities (symbolic from ModelBuilder)
        # -----------------------------------------------------------
        V_budget = self.symbolic_params.get(
            "v_cmax_tot", 220.0
        )  # Total enzyme budget (e.g., nitrogen)
        v_pmax_25 = self.symbolic_params["v_pmax"]  # PEPC
        v_cmax_mes_25 = self.symbolic_params["v_cmax_mes"]  # Rubisco in mesophyll
        v_cmax_bs_25 = self.symbolic_params["v_cmax_bs"]  # Rubisco in bundle sheath

        # Update enzyme capacities based on the fractions and total budget
        v_pmax = f_pepc * V_budget
        v_cmax_mes = f_rubisco_mes * V_budget
        v_cmax_bs = f_rubisco_bs * V_budget

        # -----------------------------------------------------------
        # Temperature dependence of Vmax values
        # -----------------------------------------------------------
        v_pmax_ea = self.symbolic_params["v_pmax_ea"]
        v_pmax_hd = self.symbolic_params["v_pmax_hd"]
        v_pmax_delta_s = self.symbolic_params["v_pmax_delta_s"]
        v_pmax = _temp_dependence(
            v_pmax, temp_mes, v_pmax_ea, v_pmax_hd, v_pmax_delta_s
        )

        v_cmax_ea = self.symbolic_params["v_cmax_ea"]
        v_cmax_hd = self.symbolic_params["v_cmax_hd"]
        v_cmax_delta_s = self.symbolic_params["v_cmax_delta_s"]
        v_cmax_mes = _temp_dependence(
            v_cmax_mes, temp_mes, v_cmax_ea, v_cmax_hd, v_cmax_delta_s
        )
        v_cmax_bs = _temp_dependence(
            v_cmax_bs, temp_bs, v_cmax_ea, v_cmax_hd, v_cmax_delta_s
        )

        # -----------------------------------------------------------
        # Kinetic parameters (temperature-corrected)
        # -----------------------------------------------------------
        k_c_25, k_o_25, k_p_25 = (
            self.symbolic_params["k_c"],
            self.symbolic_params["k_o"],
            self.symbolic_params["k_p"],
        )
        k_c_ea, k_o_ea, k_p_ea = (
            self.symbolic_params["k_c_ea"],
            self.symbolic_params["k_o_ea"],
            self.symbolic_params["k_p_ea"],
        )
        k_c_mes = _temp_dependence(k_c_25, temp_mes, k_c_ea)
        k_c_bs = _temp_dependence(k_c_25, temp_bs, k_c_ea)
        k_o_mes = _temp_dependence(k_o_25, temp_mes, k_o_ea)
        k_o_bs = _temp_dependence(k_o_25, temp_bs, k_o_ea)
        k_p = _temp_dependence(k_p_25, temp_mes, k_p_ea)

        # -----------------------------------------------------------
        # Respiration and compensation points
        # -----------------------------------------------------------
        r_ea = self.symbolic_params["r_ea"]
        r_mes_25 = self.symbolic_params["r_mes"]
        r_bs_25 = self.symbolic_params["r_bs"]
        r_mes = _temp_dependence(r_mes_25, temp_mes, r_ea)
        r_bs = _temp_dependence(r_bs_25, temp_bs, r_ea)

        gamma_star_25 = self.symbolic_params["gamma_star"]
        gamma_star_ea = self.symbolic_params["gamma_star_ea"]
        gamma_star_mes = _temp_dependence(gamma_star_25, temp_mes, gamma_star_ea)
        gamma_star_bs = _temp_dependence(gamma_star_25, temp_bs, gamma_star_ea)

        a_o_25 = self.symbolic_params["a_o"]
        a_o_ea = self.symbolic_params["a_o_ea"]
        a_o = _temp_dependence(a_o_25, temp_bs, a_o_ea)
        alpha = self.symbolic_params["alpha"]

        # -----------------------------------------------------------
        # Conductances
        # -----------------------------------------------------------
        g_m_25 = self.symbolic_params["g_m"]
        g_m_ea = self.symbolic_params["g_m_ea"]
        g_m_uc = uc.convert_conductance_w_bar(g_m_25)
        g_m = _temp_dependence(g_m_uc, temp_mes, g_m_ea)
        self.symbolic_params["g_m_uc"] = g_m

        g_bs_25 = self.symbolic_params["g_bs"]
        g_bs = uc.convert_conductance_w_bar(g_bs_25)
        self.symbolic_params["g_bs_uc"] = g_bs

        g_s_m_s = uc.convert_conductance(g_s)

        # -----------------------------------------------------------
        # Core CO₂ and enzyme fluxes
        # -----------------------------------------------------------
        co2_uptake_rate = _calculate_co2_uptake_rate(g_s_m_s, g_m, co2_atm, co2_mes)
        pepc_rate = _calculate_pepc_rate(v_pmax, k_p, co2_mes)

        co2_assimilation_rate_bs = _safe_assimilation_bs_rate(
            k_o_bs,
            k_c_bs,
            pepc_rate,
            r_mes,
            r_bs,
            g_bs,
            co2_mes,
            o2_mes,
            v_cmax_bs,
            gamma_star_bs,
            a_o,
            alpha,
        )

        co2_assimilation_rate_mes = _calculate_assimilation_mes_rate(
            k_o_mes, k_c_mes, r_mes, co2_mes, o2_mes, v_cmax_mes, gamma_star_mes
        )

        # Mass balance: CO₂ input = total assimilation
        self.opti.subject_to(
            co2_uptake_rate - co2_assimilation_rate_bs - co2_assimilation_rate_mes == 0
        )

        # -----------------------------------------------------------
        # Output dictionary (µmol m⁻² s⁻¹)
        # -----------------------------------------------------------
        self.output = {
            "v_pmax": v_pmax,
            "v_cmax_mes": v_cmax_mes,
            "v_cmax_bs": v_cmax_bs,
            "co2_assimilation_rate_bs": ca.fmax(co2_assimilation_rate_bs, 1e-9),
            "co2_assimilation_rate_mes": ca.fmax(co2_assimilation_rate_mes, 1e-9),
            "pepc_rate": ca.fmax(pepc_rate, 1e-9),
            "r_mes": r_mes,
            "r_bs": r_bs,
            "co2_uptake_rate": ca.fmax(co2_uptake_rate, 1e-9),
            "gamma_star_mes": gamma_star_mes,
            "gamma_star_bs": gamma_star_bs,
        }

    def get_output(self):
        return self.output


# ===========================================================
# Helper functions
# ===========================================================


def _temp_dependence(param_25, temp, ea, hd=None, delta_s=0):
    """Temperature response following peaked Arrhenius (if hd provided)."""
    r = 8.3145  # J mol⁻¹ K⁻¹
    if hd is None:
        return param_25 * ca.exp(ea * (temp - 298.15) / (298.15 * r * temp))
    else:
        return (
            param_25
            * ca.exp((ea * (temp - 298.15)) / (298.15 * r * temp))
            * (1 + ca.exp((298.15 * delta_s - hd) / (298.15 * r)))
            / (1 + ca.exp((temp * delta_s - hd) / (temp * r)))
        )


def _safe_assimilation_bs_rate(
    k_o,
    k_c,
    pepc_rate,
    r_mes,
    r_bs,
    g_bs,
    co2_mes,
    o2_mes,
    v_cmax_bs,
    gamma_star,
    a_o,
    alpha,
):
    """Numerically stable quadratic form for bundle sheath assimilation."""
    a = 1 - (alpha / a_o) * (k_o / k_c)
    b_c = -(
        (pepc_rate - r_mes + g_bs * co2_mes)
        + (v_cmax_bs - r_bs)
        + g_bs * (k_c * (1 + o2_mes / k_c))
        + ((alpha / a_o) * (gamma_star * v_cmax_bs + (r_bs * k_c / k_o)))
    )
    c_c = (v_cmax_bs - r_bs) * (pepc_rate - r_mes + g_bs * co2_mes) - (
        v_cmax_bs * g_bs * gamma_star * o2_mes
        + r_bs * g_bs * k_c * (1 + (o2_mes / k_o))
    )
    disc = b_c**2 - 4 * a * c_c
    disc = ca.fmax(disc, 0)  # avoid negative under sqrt
    return ca.fmax((-b_c - ca.sqrt(disc)) / (2 * a), 1e-9)


def _calculate_assimilation_mes_rate(
    k_o, k_c, r_mes, co2_mes, o2_mes, v_cmax_mes, gamma_star
):
    """Rubisco-limited mesophyll assimilation."""
    return (
        v_cmax_mes
        * (co2_mes - gamma_star * o2_mes)
        / (co2_mes + k_c * (1 + o2_mes / k_o))
        - r_mes
    )


def _calculate_co2_uptake_rate(g_s, g_m, co2_atm, co2_mes):
    return (g_s * g_m) / (g_s + g_m) * (co2_atm - co2_mes)


def _calculate_pepc_rate(v_pmax, k_p, co2_mes):
    return v_pmax * co2_mes / (k_p + co2_mes)
