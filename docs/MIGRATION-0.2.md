# Scientific and API migration to 0.2.0

Failed IPOPT solves now raise by default; diagnostic mode returns success=false and missing values. A real default solve and constraint residual are tested. The old hybrid optimizer misaligned cached evaluations and paired offspring with parent fitness. It is replaced by seeded CMA-ES ask/tell with ordered results, exact cache keys and successful-evaluation statistics. Historical optimization trajectories will change.

Use optimize_traits; the run_evolution_simulation entry point is deprecated. Selection beta, elite fraction and timeout customization are rejected. Serial execution is explicit. Parameters overwritten by internal capacity allocation are rejected as outer traits. This is optimization, not a Kimura population-genetic model. Fitness retains mol m^-2 s^-1; assimilation_total uses micromol m^-2 s^-1. The isotope value is only an illustrative interpolation proxy.

Existing heat, hydraulic and biochemical equations are preserved. The catalytic-capacity allocation is not a calibrated nitrogen budget; residual redistribution, positive capacity floors and nonsmooth maxima remain. No claim of physiological validation, SurEau equivalence, dynamic ABA control or publication readiness is made. Those require a separately specified scientific model change and empirical benchmarks.
