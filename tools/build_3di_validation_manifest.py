#!/usr/bin/env python3
"""Extract an experimental mmCIF reference panel using Foldseek 10-941cd33."""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--foldseek", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from csubst.structural_reference import extract_reference
    from csubst.structural_validation import validate_manifest
    foldseek = shutil.which(args.foldseek)
    if foldseek is None:
        parser.error("Foldseek executable not found")
    panel = json.loads(args.panel.read_text())
    if panel.get("schema_version") != 1 or not panel.get("structures"):
        parser.error("Panel requires schema_version=1 and a nonempty structures list")
    ids = [row["id"] for row in panel["structures"]]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids):
        parser.error("Panel IDs must be unique filename-safe identifiers")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for row in panel["structures"]:
        source = Path(row["path"])
        if source.is_absolute() or ".." in source.parts:
            parser.error("Panel paths must be relative to source-root without '..'")
        aa, labels, mapping, provenance = extract_reference(
            args.source_root / source, row["label_asym_id"], foldseek,
            args.output_dir / row["id"], model=row.get("model", 1))
        (args.output_dir / row["id"] / "residue_map.json").write_text(json.dumps(mapping, indent=2) + "\n")
        records.append(dict(id=row["id"], amino_acids=aa, structure_3di=labels,
                            family=row["family"], group=row["group"], split=row["split"],
                            truth_source="experimental", source=str(source), provenance=provenance,
                            residue_map=row["id"] + "/residue_map.json"))
    manifest = dict(schema_version=1, purpose="pilot_only_no_calibrator_fit", records=records)
    validate_manifest(manifest)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "panel.json").write_text(json.dumps(panel, indent=2) + "\n")
    print("Wrote", len(records), "experimental chain references to", args.output_dir)


if __name__ == "__main__":
    main()
