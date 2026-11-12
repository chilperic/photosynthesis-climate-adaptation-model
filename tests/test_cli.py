import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_reproducible_cli_and_no_overwrite(tmp_path):
    config = Path(__file__).resolve().parents[1] / "examples/reproducible_run.json"
    out = tmp_path / "run"
    command = [
        sys.executable,
        "-m",
        "photosynthesis_adaptation",
        "--config",
        str(config),
        "--output",
        str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["source_sha256"]
    for name, checksum in manifest["output_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == checksum
    assert subprocess.run(command, capture_output=True).returncode != 0
