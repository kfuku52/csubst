"""Integer text must survive parsing without a lossy float intermediate."""

import importlib

import pandas as pd
import pytest

from pymol_fakes import import_parser_pymol_with_fake_pymol


@pytest.mark.parametrize("module_name,function_name", [
    ("sequence", "_normalize_branch_ids"),
    ("parser_misc", "_normalize_branch_ids"),
    ("parser_iqtree", "_normalize_selected_branch_ids"),
    ("substitution", "_normalize_branch_ids"),
    ("parser_biodb", "_normalize_branch_ids"),
    ("foreground", "_normalize_branch_ids"),
    ("combination", "_normalize_node_ids"),
    ("output_manifest", "_normalize_branch_ids"),
    ("site_tree_plot", "_normalize_branch_ids"),
    ("parser_pymol", "_normalize_branch_ids"),
    ("parser_pymol", "_parse_positive_site"),
    ("parser_uniprot", "_parse_positive_site"),
    ("omega", "_get_cb_ids"),
])
def test_integer_identifiers_preserve_exact_text(module_name, function_name, monkeypatch):
    text = "+9223372036854775807.0"
    if module_name in {"parser_pymol", "parser_biodb"}:
        import_parser_pymol_with_fake_pymol(monkeypatch, pdb_fasta="", commands=[])
    module = importlib.import_module("csubst." + module_name)
    parse = getattr(module, function_name)
    if function_name == "_parse_positive_site":
        actual = parse(text)
    elif function_name == "_get_cb_ids":
        actual = parse(pd.DataFrame({"branch_id_1": [text]}))[0, 0]
    else:
        actual = parse([text])[0]
    assert actual == int(text.split(".", 1)[0])
