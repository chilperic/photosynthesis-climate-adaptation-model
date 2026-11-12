# Reproduce a run

The validated environment is Python 3.12 on Linux. Library lower-bound compatibility is not a claim that every permitted dependency combination was tested.

```bash
python3.12 -m venv .venv
# Windows: .venv\Scripts\activate
. .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
python -m photosynthesis_adaptation --config examples/reproducible_run.json --output runs/example-001
```

The output directory must be new. The JSON configuration, package/runtime
versions, source-file hashes and output-file hashes are recorded in `run.json`.
Keep that manifest alongside results. An example is a numerical demonstration,
not evidence of a calibrated biological model. Inspect `docs/MIGRATION-0.2.md`
before comparing with historical results. Preserve the old commit and run both
versions if a previously published calculation matters.

`requirements-lock.txt` pins the dependency resolution for Python 3.12 and
includes distribution hashes. Regenerate deliberately using
`uv pip compile pyproject.toml --extra dev --python-version 3.12 --generate-hashes -o requirements-lock.txt`,
then repeat tests and the example. It is not a cross-platform bitwise-result
guarantee: native BLAS and solver builds can differ. Compare with documented
tolerances. For CPU reproducibility set `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` before starting Python.

## Data

`data-manifest.json` records SHA-256 fingerprints of inherited inputs.
Do not overwrite inputs with fitted or cleaned data. Record transformations
as separate scripts and outputs. Check units and redistribution rights before
publishing a dataset. The software MIT license does not establish third-party
data rights.
