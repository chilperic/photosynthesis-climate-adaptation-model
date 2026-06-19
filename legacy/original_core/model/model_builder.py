import casadi as ca
import model.hydraulic_model as hydraulic
import model.heat_model as heat
import model.biochemical_model as biochemical


class ModelBuilder:
    """
    Modular CasADi-based model coupling hydraulics, heat, and biochemical processes.
    Includes total enzyme (nitrogen) budget partitioning among PEPC and Rubisco
    in both mesophyll and bundle sheath compartments (C3–C4 continuum).
    """

    def __init__(self, params):
        self.params = params
        self.symbolic_params = None
        self.evolvable_params = {}
        self.variables = None
        self.opti = None
        self.hydraulic_output = None
        self.heat_output = None
        self.biochemical_output = None
        self._rubisco_split_telemetry = {}

    # ===============================================================
    # BUILD MODEL
    # ===============================================================
    def build(self):
        self.opti = ca.Opti()
        self._initialize_parameters(self.opti)

        # --- Compute physical air temperature safely ---
        self._calculate_air_temperature()

        # ===========================================================
        # Decision variables (enzyme fractions + physical vars)
        # ===========================================================
        self.variables = {
            "g_s": self.opti.variable(),
            "temp_mes": self.opti.variable(),
            "temp_bs": self.opti.variable(),
            "co2_mes": self.opti.variable(),
            "f_pepc": self.opti.variable(),  # PEPC enzyme fraction
            "f_rubisco_mes": self.opti.variable(),  # Rubisco in mesophyll
            "f_rubisco_bs": self.opti.variable(),  # Rubisco in bundle sheath
        }

        v = self.variables

        # ===========================================================
        # Physical bounds
        # ===========================================================
        self.opti.subject_to(v["g_s"] >= 1e-7)
        self.opti.subject_to(v["temp_mes"] >= 0)
        self.opti.subject_to(v["temp_bs"] >= 0)
        self.opti.subject_to(v["co2_mes"] >= 0)

        # ===========================================================
        # Enzyme fraction bounds (biologically plausible)
        # ===========================================================
        self.opti.subject_to(v["f_pepc"] >= 0)
        self.opti.subject_to(v["f_pepc"] <= 1)  # ≤ 35% nitrogen budget
        self.opti.subject_to(v["f_rubisco_mes"] >= 0)
        self.opti.subject_to(v["f_rubisco_mes"] <= 1)
        self.opti.subject_to(v["f_rubisco_bs"] >= 0)
        self.opti.subject_to(v["f_rubisco_bs"] <= 1)  # ≤ 50% nitrogen budget

        # --- Enzyme budget conservation (≤ 1.0 for feasibility) ---
        self.opti.subject_to(
            v["f_pepc"] + v["f_rubisco_mes"] + v["f_rubisco_bs"] <= 1.0
        )

        # ===========================================================
        # Initial guesses (C3-like)
        # ===========================================================
        self.opti.set_initial(v["g_s"], 0.1)
        self.opti.set_initial(v["temp_mes"], self.params["b_air"])
        self.opti.set_initial(v["temp_bs"], self.params["b_air"])
        self.opti.set_initial(v["co2_mes"], self.params["co2_atm"])
        self.opti.set_initial(v["f_pepc"], 0.05)
        self.opti.set_initial(v["f_rubisco_mes"], 0.9)
        self.opti.set_initial(v["f_rubisco_bs"], 0.05)

        # ===========================================================
        # Enzyme (Nitrogen) budget allocation
        # ===========================================================
        V_budget = self.symbolic_params.get("v_cmax_tot", 220.0)  # Total enzyme budget
        f_pepc = v["f_pepc"]
        f_mes = v["f_rubisco_mes"]
        f_bs = v["f_rubisco_bs"]

        # Residual handling (ensures total ≤ 1 without infeasibility)
        f_sum = f_pepc + f_mes + f_bs
        f_resid = ca.fmax(
            0, 1 - f_sum
        )  # If the total is > 1, push residual to mesophyll
        f_mes = f_mes + 0.5 * f_resid  # redistribute half of residual to mesophyll
        f_pepc = ca.fmax(1e-6, f_pepc)  # Ensure PEPC fraction is not too small
        f_bs = ca.fmax(1e-6, f_bs)  # Ensure Rubisco fraction in BS is not too small

        # Symbolic catalytic capacities: V_pmax, V_cmax_mes, V_cmax_bs
        V_pmax_expr = f_pepc * V_budget
        V_cmax_mes_expr = f_mes * V_budget
        V_cmax_bs_expr = f_bs * V_budget

        # Inject symbolic expressions into parameters
        self.symbolic_params["v_pmax"] = V_pmax_expr
        self.symbolic_params["v_cmax_mes"] = V_cmax_mes_expr
        self.symbolic_params["v_cmax_bs"] = V_cmax_bs_expr

        # --- Telemetry for logging/debugging ---
        self._rubisco_split_telemetry.update(
            {
                "fraction_pepc": f_pepc,
                "fraction_rubisco_mes": f_mes,
                "fraction_rubisco_bs": f_bs,
                "v_budget_total": V_budget,
                "v_pmax_calc": V_pmax_expr,
                "v_cmax_mes_calc": V_cmax_mes_expr,
                "v_cmax_bs_calc": V_cmax_bs_expr,
            }
        )

        # ===========================================================
        # Submodels
        # ===========================================================
        htm = hydraulic.HydraulicModel(self.opti, self.symbolic_params, self.variables)
        htm.build_model()
        self.hydraulic_output = htm.get_output()

        heatm = heat.HeatModel(
            self.opti, self.symbolic_params, self.variables, self.hydraulic_output
        )
        heatm.build_model()
        self.heat_output = heatm.get_output()

        biom = biochemical.BiochemicalModel(
            self.opti, self.symbolic_params, self.variables
        )
        biom.build_model()
        self.biochemical_output = biom.get_output()

        # ===========================================================
        # Objective: maximize total assimilation (µmol m⁻² s⁻¹)
        # ===========================================================
        a_bs = self.biochemical_output.get("co2_assimilation_rate_bs", 0)
        a_mes = self.biochemical_output.get("co2_assimilation_rate_mes", 0)
        assimilation_total = (a_bs + a_mes) * 1e6
        assimilation_total = ca.fmax(assimilation_total, 1e-6)

        self.opti.minimize(-assimilation_total)

    # ===============================================================
    # SOLVE
    # ===============================================================
    def run(self, solver_options="default"):
        _set_solver_options(self.opti, solver_options)
        try:
            sol = self.opti.solve()
        except Exception as e:
            # graceful failure
            return {
                "co2_assimilation_rate_bs": 0.0,
                "co2_assimilation_rate_mes": 0.0,
                "assimilation_total": 0.0,
                "solver_status": str(e),
                "v_pmax": 0.0,
                "v_cmax_mes": 0.0,
                "v_cmax_bs": 0.0,
            }

        res = {}
        for group in [
            self.hydraulic_output,
            self.heat_output,
            self.biochemical_output,
            self.variables,
        ]:
            res.update({k: sol.value(v) for k, v in group.items()})

        res["temp_air_C"] = sol.value(self.symbolic_params["temp_C"])
        res["temp_air_K"] = sol.value(self.symbolic_params["temp_K"])

        # Telemetry (fractions and budgets)
        for k, expr in self._rubisco_split_telemetry.items():
            try:
                res[k] = float(sol.value(expr))
            except Exception:
                res[k] = float(expr)

        # ===========================================================
        # Derived catalytic capacities (for plotting and evolution)
        # ===========================================================
        try:
            V_budget_val = float(
                sol.value(self.symbolic_params.get("v_cmax_tot", 220.0))
            )
        except Exception:
            V_budget_val = 220.0

        res["v_pmax"] = float(res.get("fraction_pepc", 0.0) * V_budget_val)
        res["v_cmax_mes"] = float(res.get("fraction_rubisco_mes", 0.0) * V_budget_val)
        res["v_cmax_bs"] = float(res.get("fraction_rubisco_bs", 0.0) * V_budget_val)

        # Assimilation total (final check)
        res["assimilation_total"] = (
            res.get("co2_assimilation_rate_bs", 0.0)
            + res.get("co2_assimilation_rate_mes", 0.0)
        ) * 1e6

        return res

    # ===============================================================
    # INTERNAL HELPERS
    # ===============================================================
    def _calculate_air_temperature(self):
        """Compute air temperature from irradiance and clamp range."""
        a_air = self.symbolic_params["a_air"]
        b_air = self.symbolic_params["b_air"]
        irr = self.symbolic_params["solar_irradiance"]

        temp_C = a_air * irr + b_air
        temp_C = ca.fmax(5, ca.fmin(45, temp_C))
        temp_K = temp_C + 273.15

        self.symbolic_params["temp_C"] = temp_C
        self.symbolic_params["temp_K"] = temp_K
        self.symbolic_params["temp"] = temp_K  # for backward compatibility

    def _initialize_parameters(self, opti):
        """Convert params dict → CasADi parameters."""
        self.symbolic_params = {n: opti.parameter() for n in self.params}
        for name, sym in self.symbolic_params.items():
            opti.set_value(sym, self.params[name])

    def _set_evolvable_parameters(self, symbolic_params):
        """Register evolvable parameters."""
        evolvable_list = [
            "leaf_water_potential",
            "k_l",
            "f_canopy",
            "alpha_mes",
            "rho_mes",
            "tau_mes",
            "gamma_mes",
            "alpha_bs",
            "rho_bs",
            "tau_bs",
            "gamma_bs",
            "u_a",
            "g_h",
            "leaf_emissivity",
            "v_pmax",
            "k_p",
            "v_cmax_mes",
            "v_cmax_bs",
            "k_c",
            "k_o",
            "r_mes",
            "r_bs",
            "g_bs",
        ]
        for n, sym in symbolic_params.items():
            if n in evolvable_list:
                self.evolvable_params[n] = sym


# ===============================================================
# SOLVER OPTIONS
# ===============================================================
def _set_solver_options(opti, solver_options):
    """Quiet, robust, and faster IPOPT setup."""
    if solver_options == "default":
        opti.solver(
            "ipopt",
            {
                "ipopt": {
                    "print_level": 0,
                    "sb": "yes",
                    "max_iter": 3000,
                    "tol": 1e-6,
                    "acceptable_tol": 1e-5,
                    "acceptable_iter": 50,
                    "warm_start_init_point": "yes",
                    "mu_strategy": "adaptive",
                    "dual_inf_tol": 1e3,
                    "linear_solver": "mumps",
                    "honor_original_bounds": "yes",
                },
                "print_time": 0,
            },
        )
    else:
        raise ValueError(f"Solver options '{solver_options}' not recognized.")
