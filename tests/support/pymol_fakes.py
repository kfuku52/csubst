import importlib
import sys
import types


def import_parser_pymol_with_fake_pymol(
    monkeypatch,
    pdb_fasta,
    chains=None,
    commands=None,
    names=None,
    count_atoms=None,
):
    if chains is None:
        chains = []
    if commands is None:
        commands = []
    if names is None:
        names = []
    if count_atoms is None:
        count_atoms = {}

    def _record_do(command):
        commands.append(command)

    fake_cmd = types.SimpleNamespace(
        get_fastastr=lambda **kwargs: pdb_fasta,
        get_chains=lambda *args, **kwargs: list(chains),
        get_names=lambda *args, **kwargs: list(names),
        count_atoms=lambda selection: count_atoms.get(selection, 0),
        do=_record_do,
    )
    fake_pymol = types.SimpleNamespace(cmd=fake_cmd)
    monkeypatch.setitem(sys.modules, "pymol", fake_pymol)
    sys.modules.pop("csubst.parser_pymol", None)
    return importlib.import_module("csubst.parser_pymol")
