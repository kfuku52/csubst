"""Auditable mmCIF-chain to Foldseek reference labels for a pilot panel.

Uses the pinned Foldseek descriptor exporter to recover the chosen partner.
No predicted coordinates or reconstructed missing atoms are accepted silently.
"""

import hashlib
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from csubst.structural_validation import STATE_ORDER


FOLDSEEK_VERSION = "941cd33ff0771cd2e3f144e3293e22a2b87e9fda"
AA_NAMES = dict(zip(
    "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split(),
    STATE_ORDER,
))
AA_NAMES.update(MSE="M", UNK="X")


def read_reference_chain(path, label_asym_id, model=1):
    """Read full polymer scheme and observed backbone atoms; retain both IDs."""
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError("Reference extraction requires gemmi; install csubst[structural-validation].") from exc
    block = gemmi.cif.read(str(path)).sole_block()

    def rows(category):
        table = block.find_mmcif_category(category)
        names = [str(tag).split(".", 1)[1] for tag in table.tags]
        return [dict(zip(names, map(gemmi.cif.as_string, row))) for row in table]

    methods = [gemmi.cif.as_string(v) for v in block.find_values("_exptl.method")]
    predicted = bool(block.find_mmcif_category("_ma_model_list."))
    experimental_methods = {"X-RAY DIFFRACTION", "NEUTRON DIFFRACTION", "ELECTRON MICROSCOPY",
                            "ELECTRON CRYSTALLOGRAPHY", "SOLUTION NMR", "SOLID-STATE NMR"}
    if predicted or not methods or not set(methods) <= experimental_methods:
        raise ValueError("Pilot reference requires declared experimental coordinates, not a predicted/unknown source.")
    scheme = [row for row in rows("_pdbx_poly_seq_scheme.") if row["asym_id"] == label_asym_id]
    scheme.sort(key=lambda row: int(row["seq_id"]))
    if not scheme or [int(row["seq_id"]) for row in scheme] != list(range(1, len(scheme) + 1)):
        raise ValueError("Chain needs a unique, complete, consecutively indexed polymer scheme.")
    if any(row["mon_id"] not in AA_NAMES for row in scheme):
        raise ValueError("Unsupported polymer monomer; no silent amino-acid normalization.")
    sequence = "".join(AA_NAMES[row["mon_id"]] for row in scheme)
    atoms: dict[int, list[Any]] = {i: [] for i in range(1, len(scheme) + 1)}
    for row in rows("_atom_site."):
        if row["label_asym_id"] != label_asym_id or row.get("pdbx_PDB_model_num", "1") != str(model):
            continue
        if not row.get("label_seq_id"):
            continue
        seq_id = int(row["label_seq_id"])
        if seq_id not in atoms or row["label_comp_id"] != scheme[seq_id - 1]["mon_id"]:
            raise ValueError("Atom-site residue disagrees with the polymer scheme.")
        reference = scheme[seq_id - 1]
        if ((row.get("auth_asym_id") and row["auth_asym_id"] != reference["pdb_strand_id"])
                or (row.get("auth_seq_id") and reference.get("auth_seq_num")
                    and row["auth_seq_id"] != reference["auth_seq_num"])):
            raise ValueError("Atom-site author residue identifiers disagree with the polymer scheme.")
        if row["label_atom_id"] not in ("N", "CA", "C", "O", "CB"):
            continue
        coordinates = np.array([float(row["Cartn_" + axis]) for axis in "xyz"])
        occupancy = float(row["occupancy"])
        if not np.isfinite(coordinates).all() or not np.isfinite(occupancy) or not 0 <= occupancy <= 1:
            raise ValueError("Invalid atomic coordinates or occupancy.")
        if occupancy > 0:
            atoms[seq_id].append((row["label_atom_id"], row["label_alt_id"], occupancy, coordinates))
    mapping, observed = [], []
    for row in scheme:
        seq_id = int(row["seq_id"])
        entries = atoms[seq_id]
        alternatives = sorted({entry[1] for entry in entries if entry[1]}) or [""]
        # Highest total backbone/CB occupancy; lexical tie-breaking, common atoms retained.
        selected_alt = max(alternatives, key=lambda alt: sum(e[2] for e in entries if not e[1] or e[1] == alt))
        selected = {}
        for name, alt, occupancy, coords in entries:
            if alt and alt != selected_alt:
                continue
            if name in selected:
                raise ValueError("Duplicate atom within selected residue conformer.")
            selected[name] = coords
        item = dict(position=seq_id, monomer=row["mon_id"], amino_acid=AA_NAMES[row["mon_id"]],
                    label_asym_id=label_asym_id, auth_asym_id=row["pdb_strand_id"],
                    auth_seq_id=row.get("auth_seq_num") or None, pdb_seq_num=row.get("pdb_seq_num") or None,
                    insertion_code=row.get("pdb_ins_code") or None, model=model,
                    alt_id=selected_alt or None, atoms=sorted(selected),
                    normalization="MSE-to-M" if row["mon_id"] == "MSE" else None)
        mapping.append(item)
        if "CA" in selected:
            observed.append((seq_id, selected))
    if len(observed) < 4:
        raise ValueError("Selected chain/model has fewer than four observed CA residues.")
    return dict(sequence=sequence, mapping=mapping, observed=observed, methods=methods,
                entry_id=gemmi.cif.as_string(block.find_value("_entry.id")),
                input_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), gemmi_version=gemmi.__version__)


def write_backbone_pdb(chain, path):
    """Export selected atoms in one chain, with original polymer indices.

    PDB coordinates are rounded to 0.001 Angstrom; this exact extraction input
    is retained as an artifact. Modified MSE backbone is written as MET, and its
    descriptor neighborhood will be excluded from scoring.
    """
    lines, serial = [], 1
    for seq_id, atoms in chain["observed"]:
        if seq_id > 9999:
            raise ValueError("Pilot PDB export supports polymer indices up to 9999.")
        monomer = chain["mapping"][seq_id - 1]["monomer"]
        monomer = "MET" if monomer == "MSE" else monomer
        for name, coords in atoms.items():
            if np.any(coords <= -1000) or np.any(coords >= 10000):
                raise ValueError("Coordinate outside PDB field range.")
            lines.append(f"ATOM  {serial:5d} {name:^4s} {monomer:3s} A{seq_id:4d}    "
                         f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00 20.00          {name[0]:>2s}\n")
            serial += 1
    Path(path).write_text("".join(lines) + "TER\nEND\n", encoding="ascii")


def project_descriptors(chain, observed_aa, raw_3di, features):
    """Mask missing/undefined geometry, breaks and compressed partner distances."""
    observed = chain["observed"]
    ids = np.array([seq_id for seq_id, _ in observed])
    expected_aa = "".join(chain["sequence"][i - 1] for i in ids)
    features = np.asarray(features, dtype=float)
    if observed_aa != expected_aa or len(raw_3di) != len(ids) or not set(raw_3di) <= set(STATE_ORDER):
        raise ValueError("Foldseek sequence does not match the exported residue mapping.")
    if features.shape != (len(ids), 10) or not np.isfinite(features).all():
        raise ValueError("Foldseek descriptors must be finite residue-by-10 values.")
    labels = ["?"] * len(chain["sequence"])
    mapping = [dict(row, mask_reason="missing_CA", raw_3di=None, partner_position=None) for row in chain["mapping"]]
    for i, (seq_id, _) in enumerate(observed):
        row = mapping[seq_id - 1]
        row["raw_3di"] = raw_3di[i]
        f = features[i]
        reason = None
        partner = None
        if i in (0, len(ids) - 1) or f[7] <= 0 or f[9] == 0:
            reason = "undefined_descriptor"
        else:
            if abs(f[9]) > np.log(len(ids)) + .001:
                raise ValueError("Descriptor partner distance is outside the chain.")
            delta = int(round(np.expm1(abs(f[9])))) * (1 if f[9] > 0 else -1)
            partner = i + delta
            if not 0 < partner < len(ids) - 1 or delta == 0 or not np.isclose(f[8], np.clip(delta, -4, 4), atol=.001):
                raise ValueError("Descriptor partner index is inconsistent.")
            row["partner_position"] = int(ids[partner])
            neighborhoods = [i - 1, i, i + 1, partner - 1, partner, partner + 1]
            if any(not {"N", "CA", "C"} <= set(observed[j][1]) for j in neighborhoods):
                reason = "missing_backbone"
            elif any(mapping[ids[j] - 1]["monomer"] in ("MSE", "UNK") for j in neighborhoods):
                reason = "modified_or_unknown_neighborhood"
            elif abs(int(ids[partner] - ids[i])) != abs(delta):
                reason = "compressed_partner_distance"
            else:
                for center in (i, partner):
                    for left, right in ((center - 1, center), (center, center + 1)):
                        distance = np.linalg.norm(observed[left][1]["CA"] - observed[right][1]["CA"])
                        if ids[right] - ids[left] != 1 or not 2 <= distance <= 4.5:
                            reason = "chain_break"
        row["mask_reason"] = reason
        if reason is None:
            labels[seq_id - 1] = raw_3di[i]
    return "".join(labels), mapping


def extract_reference(path, label_asym_id, foldseek, output_dir, model=1):
    """Extract one experimental chain and retain exact PDB/descriptors/provenance."""
    version = subprocess.check_output([str(foldseek), "version"], text=True).strip()
    if version != FOLDSEEK_VERSION:
        raise ValueError("Reference extraction requires Foldseek release 10-941cd33 (descriptor schema is pinned).")
    chain = read_reference_chain(path, label_asym_id, model=model)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    pdb = out / "reference.pdb"
    write_backbone_pdb(chain, pdb)
    descriptor = out / "descriptors"
    command = [str(foldseek), "structureto3didescriptor", str(pdb), str(descriptor), "--threads", "1"]
    run = subprocess.run(command, capture_output=True, text=True)
    (out / "foldseek.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError("Foldseek reference extraction failed; inspect its log.")
    text = descriptor.read_bytes().rstrip(b"\x00\n").decode()
    fields = text.split("\t")
    if len(fields) != 4:
        raise ValueError("Expected exactly one Foldseek descriptor record.")
    numbers = np.array([float(v) for v in fields[3].strip().split(",")])
    if numbers.size != len(fields[1]) * 10:
        raise ValueError("Unexpected Foldseek descriptor length.")
    labels, mapping = project_descriptors(chain, fields[1], fields[2], numbers.reshape(-1, 10))
    provenance = dict(entry_id=chain["entry_id"], methods=chain["methods"], input_sha256=chain["input_sha256"],
                      gemmi_version=chain["gemmi_version"], foldseek_version=version,
                      foldseek_binary_sha256=hashlib.sha256(Path(foldseek).read_bytes()).hexdigest(),
                      export_sha256=hashlib.sha256(pdb.read_bytes()).hexdigest(),
                      coordinate_precision_angstrom=.001, chain_break_ca_range=[2, 4.5],
                      protocol="experimental-chain-conservative-mask-v1")
    return chain["sequence"], labels, mapping, provenance
