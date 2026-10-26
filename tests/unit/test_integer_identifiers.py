"""Integer text must survive parsing without a lossy float intermediate."""

import importlib

import pytest

from pymol_fakes import import_parser_pymol_with_fake_pymol


@pytest.mark.parametrize("module_name", ["parser_pymol", "parser_uniprot"])
def test_external_site_identifiers_preserve_exact_text(module_name, monkeypatch):
    text = "+9223372036854775807.0"
    if module_name == "parser_pymol":
        import_parser_pymol_with_fake_pymol(monkeypatch, pdb_fasta="", commands=[])
    module = importlib.import_module("csubst." + module_name)
    actual = module._parse_positive_site(text)
    assert actual == int(text.split(".", 1)[0])
