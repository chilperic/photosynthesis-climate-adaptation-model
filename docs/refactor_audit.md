# Refactor audit

## Problems in the uploaded working directories

- Generated files were mixed with source code (`__pycache__`, logs, plots, and solver outputs).
- Several notebooks represented exploratory states rather than minimal reproducible workflows.
- Imports were working-directory dependent (`model.*`, `utils.*`).
- Parameter loading assumed a local `resources/` folder.
- External API access and local JSON editing were mixed into parameter-loading utilities.
- Multiple historical steady-state model folders were present without a single execution pathway.
- Lab-rotation material contains another person's name and should be treated as supervised/provenance material, not as sole-authored portfolio code unless attribution and consent are clear.

## Refactor decisions

- Core reusable code was moved into a package under `src/photosynthesis_adaptation`.
- Parameter JSON files are bundled as package resources.
- Minimal scripts were added for preset listing, smoke-model execution, short evolution, and manifest summaries.
- Original core Python files and parameter resources are preserved under `legacy/original_core`.
- Historical notebooks and steady-state materials are preserved as legacy/provenance material rather than mixed with the main package.
- Docker and CI configuration were added for reproducibility.
