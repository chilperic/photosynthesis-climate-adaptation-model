#!/usr/bin/env python
"""Run a minimal coupled photosynthesis model smoke test.

This script is intentionally small. It verifies that bundled parameters can be
loaded, the CasADi model can be built, and the solver returns a result object.
"""

import json
from pathlib import Path

from photosynthesis_adaptation.parameters import load_parameters
from photosynthesis_adaptation.model import ModelBuilder


def main() -> None:
    params = load_parameters(environment="default")
    model = ModelBuilder(params)
    model.build()
    result = model.run()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "smoke_model_result.json"
    serializable = {
        key: float(value) if isinstance(value, (int, float)) else str(value)
        for key, value in result.items()
    }
    out_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    print("success:", result.get("assimilation_total", 0.0) >= 0.0)
    print("assimilation_total:", result.get("assimilation_total"))
    print("result_file:", out_file)


if __name__ == "__main__":
    main()
