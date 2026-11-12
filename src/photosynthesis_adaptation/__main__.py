"""Run an explicit JSON configuration and export results with provenance."""

import argparse
import json
from pathlib import Path

from .provenance import write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output.exists():
        parser.error("Output path already exists; choose a new run directory")
    from .model.model_builder import ModelBuilder
    from .parameters import load_parameters

    params = load_parameters(**config.get("presets", {}))
    overrides = config.get("parameters", {})
    if set(overrides) - (set(params) | {"v_cmax_tot"}):
        raise ValueError("Unknown model parameter override")
    params.update(overrides)
    config["resolved_parameters"] = params
    model = ModelBuilder(params)
    model.build()
    result = model.run(config.get("solver", "default"))
    args.output.mkdir(parents=True)
    result = {k: v.item() if hasattr(v, "item") else v for k, v in result.items()}
    (args.output / "solution.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_manifest(
        args.output,
        config,
        args.config,
        "photosynthesis-adaptation",
        ["numpy", "scipy", "casadi", "cma"],
    )


if __name__ == "__main__":
    main()
