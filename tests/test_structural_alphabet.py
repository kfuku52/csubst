import numpy as np
import pytest

from csubst import ete
from csubst import structural_alphabet
from csubst import tree


def _build_test_tree():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    node_by_name = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    return tr, node_by_name


def _build_state_pep(node_count, aa_orders, seq_by_node, float_tol=1e-12):
    state_pep = np.zeros((node_count, 3, len(aa_orders)), dtype=float)
    aa_lookup = {aa: i for i, aa in enumerate(aa_orders.tolist())}
    for node_id, seq in seq_by_node.items():
        for site, aa in enumerate(seq):
            if aa == "-":
                continue
            state_pep[node_id, site, aa_lookup[aa]] = 1.0
    return state_pep


def test_build_3di_state_from_state_pep_projects_gaps_and_one_hot():
    tr, node_by_name = _build_test_tree()
    aa_orders = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    seq_by_node = {
        node_by_name["A"]: "AC-",
        node_by_name["B"]: "AD-",
    }
    state_pep = _build_state_pep(
        node_count=len(list(tr.traverse())),
        aa_orders=aa_orders,
        seq_by_node=seq_by_node,
    )
    g = {
        "tree": tr,
        "amino_acid_orders": aa_orders,
        "float_tol": 1e-12,
    }

    def _fake_predictor(aa_sequences, _g):
        out = dict()
        for key, seq in aa_sequences.items():
            out[key] = "VV" if len(seq) == 2 else ("V" * len(seq))
        return out

    state_3di, state_orders, aligned = structural_alphabet.build_3di_state_from_state_pep(
        g=g,
        state_pep=state_pep,
        predictor=_fake_predictor,
    )
    assert state_3di.shape == (len(list(tr.traverse())), 3, 20)
    assert state_orders.tolist() == list("ACDEFGHIKLMNPQRSTVWY")
    assert aligned[node_by_name["A"]] == "VV-"
    assert aligned[node_by_name["B"]] == "VV-"
    v_index = state_orders.tolist().index("V")
    assert state_3di[node_by_name["A"], 0, v_index] == pytest.approx(1.0)
    assert state_3di[node_by_name["A"], 1, v_index] == pytest.approx(1.0)
    assert state_3di[node_by_name["A"], 2, :].sum() == pytest.approx(0.0)


def test_build_3di_state_from_state_pep_raises_on_length_mismatch():
    tr, node_by_name = _build_test_tree()
    aa_orders = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    seq_by_node = {
        node_by_name["A"]: "AC-",
        node_by_name["B"]: "AD-",
    }
    state_pep = _build_state_pep(
        node_count=len(list(tr.traverse())),
        aa_orders=aa_orders,
        seq_by_node=seq_by_node,
    )
    g = {
        "tree": tr,
        "amino_acid_orders": aa_orders,
        "float_tol": 1e-12,
    }

    def _bad_predictor(aa_sequences, _g):
        out = dict()
        for key, seq in aa_sequences.items():
            out[key] = "V" * (len(seq) + 1)
        return out

    with pytest.raises(ValueError, match="length mismatch"):
        structural_alphabet.build_3di_state_from_state_pep(
            g=g,
            state_pep=state_pep,
            predictor=_bad_predictor,
        )


def test_build_3di_state_from_state_pep_includes_root_when_selected():
    tr, node_by_name = _build_test_tree()
    root_id = node_by_name["R"]
    tip_a_id = node_by_name["A"]
    aa_orders = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    seq_by_node = {
        root_id: "AC-",
        tip_a_id: "AC-",
    }
    state_pep = _build_state_pep(
        node_count=len(list(tr.traverse())),
        aa_orders=aa_orders,
        seq_by_node=seq_by_node,
    )
    g = {
        "tree": tr,
        "amino_acid_orders": aa_orders,
        "float_tol": 1e-12,
    }

    def _fake_predictor(aa_sequences, _g):
        return {key: ("V" * len(seq)) for key, seq in aa_sequences.items()}

    state_3di, state_orders, aligned = structural_alphabet.build_3di_state_from_state_pep(
        g=g,
        state_pep=state_pep,
        selected_branch_ids=np.array([root_id, tip_a_id], dtype=np.int64),
        predictor=_fake_predictor,
    )
    v_index = state_orders.tolist().index("V")
    assert root_id in aligned
    assert tip_a_id in aligned
    assert state_3di[root_id, 0, v_index] == pytest.approx(1.0)
    assert state_3di[root_id, 1, v_index] == pytest.approx(1.0)


def test_build_tip_aa_and_3di_alignment_from_full_cds(tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    aln_path = tmp_path / "full_cds.fa"
    aln_path.write_text(">A\nAAA---AAG\n>B\nGCT---NNN\n", encoding="utf-8")
    g = {
        "tree": tr,
        "full_cds_alignment_file": str(aln_path),
        "codon_table": [
            ("K", "AAA"),
            ("K", "AAG"),
            ("A", "GCT"),
        ],
    }
    aa = structural_alphabet.build_tip_aa_alignment_from_full_cds(g=g)
    assert aa["A"] == "K-K"
    assert aa["B"] == "A-X"

    def _fake_predictor(aa_sequences, _g):
        return {name: ("V" * len(seq)) for name, seq in aa_sequences.items()}

    out_path = tmp_path / "tip_3di.fa"
    threed = structural_alphabet.build_tip_3di_alignment_from_full_cds(
        g=g,
        predictor=_fake_predictor,
        output_path=str(out_path),
    )
    assert threed["A"] == "V-V"
    assert threed["B"] == "V-V"
    text = out_path.read_text(encoding="utf-8")
    assert ">A" in text
    assert "V-V" in text


def test_resolve_prostt5_model_source_prefers_local_dir(tmp_path):
    g = {
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": str(tmp_path),
        "prostt5_no_download": True,
    }
    source, no_download = structural_alphabet._resolve_prostt5_model_source(g=g)
    assert source == str(tmp_path)
    assert no_download is True


def test_resolve_prostt5_model_source_uses_model_name_by_default():
    g = {
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": "",
        "prostt5_no_download": False,
    }
    source, no_download = structural_alphabet._resolve_prostt5_model_source(g=g)
    assert source == "Rostlab/ProstT5"
    assert no_download is False


def test_resolve_prostt5_model_source_rejects_missing_local_dir():
    g = {
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": "/path/that/does/not/exist",
        "prostt5_no_download": True,
    }
    with pytest.raises(ValueError, match="prostt5_local_dir"):
        structural_alphabet._resolve_prostt5_model_source(g=g)


class _FakeTokenizer:
    local_sources = set()
    local_only_calls = []
    download_calls = []
    save_calls = []

    def __init__(self, source):
        self.source = str(source)

    @classmethod
    def reset(cls):
        cls.local_sources = set()
        cls.local_only_calls = []
        cls.download_calls = []
        cls.save_calls = []

    @classmethod
    def from_pretrained(cls, source, do_lower_case=False, local_files_only=False):
        source = str(source)
        cls.local_only_calls.append((source, bool(local_files_only)))
        if local_files_only:
            if source not in cls.local_sources:
                raise OSError("missing local tokenizer files")
            return cls(source=source)
        cls.download_calls.append(source)
        return cls(source=source)

    def save_pretrained(self, save_directory):
        save_directory = str(save_directory)
        type(self).save_calls.append(save_directory)
        type(self).local_sources.add(save_directory)


class _FakeModel:
    local_sources = set()
    local_only_calls = []
    download_calls = []
    save_calls = []

    def __init__(self, source):
        self.source = str(source)

    @classmethod
    def reset(cls):
        cls.local_sources = set()
        cls.local_only_calls = []
        cls.download_calls = []
        cls.save_calls = []

    @classmethod
    def from_pretrained(cls, source, local_files_only=False):
        source = str(source)
        cls.local_only_calls.append((source, bool(local_files_only)))
        if local_files_only:
            if source not in cls.local_sources:
                raise OSError("missing local model files")
            return cls(source=source)
        cls.download_calls.append(source)
        return cls(source=source)

    def save_pretrained(self, save_directory):
        save_directory = str(save_directory)
        type(self).save_calls.append(save_directory)
        type(self).local_sources.add(save_directory)


def test_ensure_prostt5_model_files_downloads_into_local_dir_when_missing(tmp_path):
    _FakeTokenizer.reset()
    _FakeModel.reset()
    g = {
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": str(tmp_path),
        "prostt5_no_download": False,
    }
    model_source = structural_alphabet.ensure_prostt5_model_files(
        g=g,
        tokenizer_cls=_FakeTokenizer,
        model_cls=_FakeModel,
    )
    assert model_source == str(tmp_path)
    assert _FakeTokenizer.download_calls == ["Rostlab/ProstT5"]
    assert _FakeModel.download_calls == ["Rostlab/ProstT5"]
    assert str(tmp_path) in _FakeTokenizer.save_calls
    assert str(tmp_path) in _FakeModel.save_calls


def test_ensure_prostt5_model_files_respects_no_download_for_missing_local_dir(tmp_path):
    _FakeTokenizer.reset()
    _FakeModel.reset()
    g = {
        "prostt5_model": "Rostlab/ProstT5",
        "prostt5_local_dir": str(tmp_path),
        "prostt5_no_download": True,
    }
    with pytest.raises(RuntimeError, match="prostt5_local_dir"):
        structural_alphabet.ensure_prostt5_model_files(
            g=g,
            tokenizer_cls=_FakeTokenizer,
            model_cls=_FakeModel,
        )
