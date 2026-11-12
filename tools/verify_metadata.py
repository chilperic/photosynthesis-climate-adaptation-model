"""Check local metadata identity and inherited data integrity (not FAIR certification)."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
meta = json.loads((root / "codemeta.json").read_text())
assert meta["@type"] == "SoftwareSourceCode"
assert meta["version"] == "0.2.0"
assert meta["codeRepository"].startswith("https://github.com/chilperic/")
manifest = json.loads((root / "data-manifest.json").read_text())
for item in manifest["files"]:
    path = (root / item["path"]).resolve()
    assert path.is_relative_to(root)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]
crate = json.loads((root / "ro-crate-metadata.json").read_text())
assert crate["@context"] == "https://w3id.org/ro/crate/1.3/context"
entities = {e["@id"]: e for e in crate["@graph"]}
assert len(entities) == len(crate["@graph"])
assert entities["./"]["@type"] == "Dataset"
for item in entities["./"]["hasPart"]:
    assert item["@id"] in entities
    assert (root / item["@id"]).is_file()
assert (root / "CITATION.cff").is_file()
print(f"Verified metadata and {len(manifest['files'])} inherited input hashes")
