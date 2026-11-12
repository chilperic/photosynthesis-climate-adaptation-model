# Architecture

The repository is organized as a reproducible modelling pipeline.

```mermaid
flowchart TD
    A[Parameter presets / environmental inputs] --> B[Parameter loader]
    B --> C[Hydraulic submodel]
    B --> D[Heat-balance submodel]
    B --> E[Biochemical photosynthesis submodel]
    C --> F[Coupled CasADi optimization model]
    D --> F
    E --> F
    F --> G[Model outputs]
    G --> H[Sensitivity analysis]
    G --> I[Evolutionary trait simulation]
    H --> J[Diagnostics and plots]
    I --> J
```

## Components

| Layer | Module | Role |
|---|---|---|
| Parameters | `photosynthesis_adaptation.parameters` | Loads bundled JSON presets from package resources. |
| Hydraulics | `model/hydraulic_model.py` | Computes transpiration and water-supply constraints. |
| Heat balance | `model/heat_model.py` | Computes mesophyll and bundle-sheath temperature constraints. |
| Biochemistry | `model/biochemical_model.py` | Computes PEPC/Rubisco assimilation and temperature-dependent kinetics. |
| Coupling | `model/model_builder.py` | Builds and solves the coupled CasADi optimization problem. |
| Sensitivity | `utils/sensitivity_analysis.py` | Local and global sensitivity workflows. |
| Evolution | `utils/evolution.py` | Seeded CMA-ES optimization; not a population-genetic evolutionary model. |
| Scripts | `scripts/` | Minimal reproducible entry points. |
| Legacy | `legacy/` | Preserved manifests and selected original material. |
