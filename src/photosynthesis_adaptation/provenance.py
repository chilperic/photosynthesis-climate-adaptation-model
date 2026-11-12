"""Portable, explicit run provenance with no hidden network or shell calls."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def write_manifest(output_dir, config, input_path, package_name, dependencies):
    output_dir = Path(output_dir)
    source = Path(__file__).parent
    source_hashes = {
        p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".json", ".asc")
    }
    versions = {name: importlib.metadata.version(name) for name in dependencies}
    versions[package_name] = importlib.metadata.version(package_name)
    outputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output_dir.iterdir())
        if p.is_file() and p.name != "run.json"
    }
    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "software": versions,
        "configuration": config,
        "configuration_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "output_sha256": outputs,
    }
    (output_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest
