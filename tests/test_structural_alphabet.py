import os
import numpy as np
import pytest
from collections import OrderedDict
from types import SimpleNamespace

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


def test_encode_tip_3di_alignment_for_morph_maps_states_and_gaps():
    tip_3di = {
        "A": "ACD-",
        "B": "MNPY",
    }
    encoded = structural_alphabet._encode_tip_3di_alignment_for_morph(
        tip_3di_by_name=tip_3di,
        output_path=None,
    )
    assert encoded["A"] == "012-"
    assert encoded["B"] == "ABCJ"


def test_normalize_direct_iqtree_model_maps_gtr20_to_gtr():
    model, remapped = structural_alphabet._normalize_direct_iqtree_model("GTR20")
    assert model == "GTR"
    assert remapped is True
    model_ok, remapped_ok = structural_alphabet._normalize_direct_iqtree_model("MK")
    assert model_ok == "MK"
    assert remapped_ok is False


def test_read_direct_3di_state_tensor_accepts_morph_state_columns(tmp_path):
    rooted_tree = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    treefile_path = tmp_path / "direct.treefile"
    tree.write_tree(rooted_tree, outfile=str(treefile_path), add_numerical_label=False)
    direct_tree = tree.standardize_node_names(ete.PhyloNode(treefile_path.read_text(), format=1))
    root_name = [n.name for n in direct_tree.traverse() if ete.is_root(n)][0]
    state_path = tmp_path / "direct.state"
    state_path.write_text(
        "\n".join(
            [
                "Node\tSite\tState\tp_0\tp_1",
                "{}\t1\t1\t0.10\t0.90".format(root_name),
                "{}\t2\t0\t0.80\t0.20".format(root_name),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    g = {
        "rooted_tree": rooted_tree,
        "float_type": np.float64,
        "ml_anc": False,
    }
    tip_3di_by_name = {"A": "AC", "B": "CA"}
    paths = {
        "treefile": str(treefile_path),
        "state": str(state_path),
        "state_symbol_mode": "morph",
    }
    state_tensor, state_orders = structural_alphabet._read_direct_3di_state_tensor(
        g=g,
        paths=paths,
        tip_3di_by_name=tip_3di_by_name,
        selected_branch_ids=None,
    )
    node_by_name = {node.name: int(ete.get_prop(node, "numerical_label")) for node in rooted_tree.traverse()}
    root_id = node_by_name["R"]
    tip_a_id = node_by_name["A"]
    idx_a = state_orders.tolist().index("A")
    idx_c = state_orders.tolist().index("C")
    assert state_tensor[root_id, 0, idx_a] == pytest.approx(0.10)
    assert state_tensor[root_id, 0, idx_c] == pytest.approx(0.90)
    assert state_tensor[root_id, 1, idx_a] == pytest.approx(0.80)
    assert state_tensor[root_id, 1, idx_c] == pytest.approx(0.20)
    assert state_tensor[tip_a_id, 0, idx_a] == pytest.approx(1.0)
    assert state_tensor[tip_a_id, 1, idx_c] == pytest.approx(1.0)


def test_run_iqtree_direct_3di_uses_morph_and_remaps_gtr20(tmp_path, monkeypatch):
    rooted_tree = ete.PhyloNode("(A:1,B:1)R;", format=1)
    tip_alignment = tmp_path / "csubst_alignment_3di_tip_morph.fa"
    tip_alignment.write_text(">A\n0\n>B\n1\n", encoding="utf-8")
    prefix = str(tip_alignment.resolve())
    captured = {"command": None}

    def _fake_run(command, stdout=None, stderr=None):
        captured["command"] = list(command)
        for ext in ["treefile", "state", "iqtree", "log"]:
            (tmp_path / ("csubst_alignment_3di_tip_morph.fa.{}".format(ext))).write_text(
                "stub\n", encoding="utf-8"
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(structural_alphabet.subprocess, "run", _fake_run)
    g = {
        "rooted_tree": rooted_tree,
        "iqtree_exe": "iqtree",
        "threads": 2,
        "iqtree_redo": True,
        "sa_iqtree_model": "GTR20",
    }
    paths = structural_alphabet._run_iqtree_direct_3di(g=g, tip_alignment_path=prefix)
    command = captured["command"]
    assert command is not None
    seqtype_index = command.index("--seqtype")
    model_index = command.index("-m")
    assert command[seqtype_index + 1] == "MORPH"
    assert command[model_index + 1] == "GTR"
    assert paths["state_symbol_mode"] == "morph"


def test_build_3di_state_direct_prefilters_tip_invariant_sites_when_enabled(monkeypatch):
    tip_full = OrderedDict([("A", "AAC"), ("B", "ABC"), ("C", "A-C")])
    captured = {"encoded_tip": None}
    num_node = 3
    num_state = 20
    reduced_tensor = np.zeros((num_node, 1, num_state), dtype=float)
    reduced_tensor[:, :, 0] = 0.25

    monkeypatch.setattr(
        structural_alphabet,
        "build_tip_3di_alignment_from_full_cds",
        lambda g, predictor=None, output_path=None: tip_full,
    )

    def _fake_encode(tip_3di_by_name, output_path=None):
        captured["encoded_tip"] = OrderedDict(tip_3di_by_name)
        return tip_3di_by_name

    monkeypatch.setattr(structural_alphabet, "_encode_tip_3di_alignment_for_morph", _fake_encode)
    monkeypatch.setattr(
        structural_alphabet,
        "_run_iqtree_direct_3di",
        lambda g, tip_alignment_path: {"treefile": "x.tree", "state": "x.state", "state_symbol_mode": "morph"},
    )
    monkeypatch.setattr(
        structural_alphabet,
        "_read_direct_3di_state_tensor",
        lambda g, paths, tip_3di_by_name, selected_branch_ids=None: (
            reduced_tensor.copy(),
            np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object),
        ),
    )

    g = {
        "drop_invariant_tip_sites": True,
        "drop_invariant_tip_sites_mode": "tip_invariant",
    }
    state_tensor, state_orders, tip_out = structural_alphabet.build_3di_state_direct(g=g)
    assert tip_out == tip_full
    assert captured["encoded_tip"] == OrderedDict([("A", "A"), ("B", "B"), ("C", "-")])
    assert "_precomputed_tip_invariant_site_mask" in g
    np.testing.assert_array_equal(g["_precomputed_tip_invariant_site_mask"], np.array([True, False, True]))
    assert state_tensor.shape == (num_node, 3, num_state)
    np.testing.assert_allclose(state_tensor[:, 0, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(state_tensor[:, 1, 0], 0.25, atol=1e-12)
    np.testing.assert_allclose(state_tensor[:, 2, :], 0.0, atol=1e-12)
    assert state_orders.tolist() == list("ACDEFGHIKLMNPQRSTVWY")


def test_build_3di_state_direct_does_not_prefilter_when_mode_is_zero_sub_mass(monkeypatch):
    tip_full = OrderedDict([("A", "AAA"), ("B", "AAA")])
    captured = {"encoded_tip": None}
    monkeypatch.setattr(
        structural_alphabet,
        "build_tip_3di_alignment_from_full_cds",
        lambda g, predictor=None, output_path=None: tip_full,
    )

    def _fake_encode(tip_3di_by_name, output_path=None):
        captured["encoded_tip"] = OrderedDict(tip_3di_by_name)
        return tip_3di_by_name

    monkeypatch.setattr(structural_alphabet, "_encode_tip_3di_alignment_for_morph", _fake_encode)
    monkeypatch.setattr(
        structural_alphabet,
        "_run_iqtree_direct_3di",
        lambda g, tip_alignment_path: {"treefile": "x.tree", "state": "x.state", "state_symbol_mode": "morph"},
    )
    monkeypatch.setattr(
        structural_alphabet,
        "_read_direct_3di_state_tensor",
        lambda g, paths, tip_3di_by_name, selected_branch_ids=None: (
            np.ones((3, 3, 20), dtype=float),
            np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object),
        ),
    )
    g = {
        "drop_invariant_tip_sites": True,
        "drop_invariant_tip_sites_mode": "zero_sub_mass",
    }
    structural_alphabet.build_3di_state_direct(g=g)
    assert captured["encoded_tip"] == tip_full
    assert "_precomputed_tip_invariant_site_mask" not in g


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


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorchRuntime:
    @staticmethod
    def no_grad():
        return _FakeNoGrad()


class _FakeBatchTensor:
    def __init__(self, payload):
        self.payload = payload

    def to(self, _device):
        return self


class _FakeBatchTokenizer:
    decode_calls = []

    @classmethod
    def reset(cls):
        cls.decode_calls = []

    def __call__(self, prompts, return_tensors="pt", padding=True):
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)
        return {
            "input_ids": _FakeBatchTensor(prompt_list),
            "attention_mask": _FakeBatchTensor([1] * len(prompt_list)),
        }

    def decode(self, pred_id, skip_special_tokens=True):
        type(self).decode_calls.append(pred_id)
        return str(pred_id["pred"])


class _FakeBatchTokenizerWithBatchDecode(_FakeBatchTokenizer):
    batch_decode_calls = []

    @classmethod
    def reset(cls):
        cls.decode_calls = []
        cls.batch_decode_calls = []

    def batch_decode(self, pred_ids, skip_special_tokens=True):
        pred_ids = list(pred_ids)
        type(self).batch_decode_calls.append(len(pred_ids))
        return [str(pred_id["pred"]) for pred_id in pred_ids]


class _FakeBatchModel:
    generate_batch_sizes = []

    @classmethod
    def reset(cls):
        cls.generate_batch_sizes = []

    def generate(
        self,
        input_ids,
        attention_mask=None,
        num_beams=1,
        do_sample=False,
        min_new_tokens=0,
        max_new_tokens=0,
    ):
        prompts = list(input_ids.payload)
        type(self).generate_batch_sizes.append(len(prompts))
        return [{"pred": "A" * int(max_new_tokens)} for _ in prompts]


def test_predict_3di_with_prostt5_batches_and_reuses_duplicates(monkeypatch):
    tokenizer = _FakeBatchTokenizer()
    model = _FakeBatchModel()
    _FakeBatchTokenizer.reset()
    _FakeBatchModel.reset()

    monkeypatch.setattr(
        structural_alphabet,
        "_load_prostt5_components",
        lambda g: (_FakeTorchRuntime(), tokenizer, model, "cpu"),
    )
    aa_sequences = {
        "n1": "AC",
        "n2": "AC",
        "n3": "XX",
        "n4": "MNP",
        "n5": "",
        "n6": "--",
    }
    out = structural_alphabet.predict_3di_with_prostt5(
        aa_sequences=aa_sequences,
        g={"threads": 2, "prostt5_cache": False},
    )
    assert out["n1"] == "AA"
    assert out["n2"] == "AA"
    assert out["n3"] == "AA"
    assert out["n4"] == "AAA"
    assert out["n5"] == ""
    assert out["n6"] == ""
    assert _FakeBatchModel.generate_batch_sizes == [2, 1]
    assert len(_FakeBatchTokenizer.decode_calls) == 3


def test_resolve_prostt5_auto_batch_size_uses_threads_by_default():
    batch_size = structural_alphabet._resolve_prostt5_auto_batch_size(
        threads=3,
        device="cpu",
        unique_sequence_count=100,
    )
    assert batch_size == 3


def test_resolve_prostt5_auto_batch_size_can_expand_on_cuda():
    batch_size = structural_alphabet._resolve_prostt5_auto_batch_size(
        threads=4,
        device="cuda",
        unique_sequence_count=100,
    )
    assert batch_size == 32


def test_resolve_prostt5_auto_batch_size_can_expand_on_mps():
    batch_size = structural_alphabet._resolve_prostt5_auto_batch_size(
        threads=2,
        device="mps",
        unique_sequence_count=100,
    )
    assert batch_size == 16


def test_predict_3di_with_prostt5_uses_cache_without_loading_model(tmp_path, monkeypatch):
    cache_path = tmp_path / "prostt5_cache.tsv"
    cache_path.write_text(
        "Rostlab/ProstT5\tAC\tAA\nRostlab/ProstT5\tXX\tAA\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        structural_alphabet,
        "_load_prostt5_components",
        lambda g: (_ for _ in ()).throw(AssertionError("model should not be loaded on full cache hit")),
    )
    out = structural_alphabet.predict_3di_with_prostt5(
        aa_sequences={"n1": "AC", "n2": "XX"},
        g={"prostt5_cache": True, "prostt5_cache_file": str(cache_path), "threads": 1},
    )
    assert out["n1"] == "AA"
    assert out["n2"] == "AA"


def test_predict_3di_with_prostt5_appends_new_cache_entries(tmp_path, monkeypatch):
    cache_path = tmp_path / "prostt5_cache.tsv"
    tokenizer = _FakeBatchTokenizer()
    model = _FakeBatchModel()
    _FakeBatchTokenizer.reset()
    _FakeBatchModel.reset()
    monkeypatch.setattr(
        structural_alphabet,
        "_load_prostt5_components",
        lambda g: (_FakeTorchRuntime(), tokenizer, model, "cpu"),
    )
    out = structural_alphabet.predict_3di_with_prostt5(
        aa_sequences={"n1": "AC", "n2": "MNP"},
        g={"prostt5_cache": True, "prostt5_cache_file": str(cache_path), "threads": 1},
    )
    assert out["n1"] == "AA"
    assert out["n2"] == "AAA"
    cache_txt = cache_path.read_text(encoding="utf-8")
    assert "Rostlab/ProstT5\tAC\tAA" in cache_txt
    assert "Rostlab/ProstT5\tMNP\tAAA" in cache_txt


def test_predict_3di_with_prostt5_prefers_batch_decode_when_available(monkeypatch):
    tokenizer = _FakeBatchTokenizerWithBatchDecode()
    model = _FakeBatchModel()
    _FakeBatchTokenizerWithBatchDecode.reset()
    _FakeBatchModel.reset()
    monkeypatch.setattr(
        structural_alphabet,
        "_load_prostt5_components",
        lambda g: (_FakeTorchRuntime(), tokenizer, model, "cpu"),
    )
    out = structural_alphabet.predict_3di_with_prostt5(
        aa_sequences={"n1": "AC", "n2": "AD"},
        g={"threads": 2, "prostt5_cache": False},
    )
    assert out["n1"] == "AA"
    assert out["n2"] == "AA"
    assert _FakeBatchTokenizerWithBatchDecode.batch_decode_calls == [2]
    assert _FakeBatchTokenizerWithBatchDecode.decode_calls == []


def _fake_torch_module(cuda_available=False, mps_available=False, mps_built=True):
    cuda_ns = SimpleNamespace(is_available=lambda: bool(cuda_available))
    mps_ns = SimpleNamespace(
        is_available=lambda: bool(mps_available),
        is_built=lambda: bool(mps_built),
    )
    return SimpleNamespace(cuda=cuda_ns, backends=SimpleNamespace(mps=mps_ns))


def test_resolve_prostt5_device_auto_prefers_cuda_then_mps():
    dev_cuda = structural_alphabet._resolve_prostt5_device(
        torch_module=_fake_torch_module(cuda_available=True, mps_available=True),
        device_opt="auto",
    )
    dev_mps = structural_alphabet._resolve_prostt5_device(
        torch_module=_fake_torch_module(cuda_available=False, mps_available=True),
        device_opt="auto",
    )
    dev_cpu = structural_alphabet._resolve_prostt5_device(
        torch_module=_fake_torch_module(cuda_available=False, mps_available=False),
        device_opt="auto",
    )
    assert dev_cuda == "cuda"
    assert dev_mps == "mps"
    assert dev_cpu == "cpu"


def test_resolve_prostt5_device_explicit_mps_requires_backend():
    with pytest.raises(ValueError, match="MPS is not available"):
        structural_alphabet._resolve_prostt5_device(
            torch_module=_fake_torch_module(cuda_available=False, mps_available=False),
            device_opt="mps",
        )
    dev = structural_alphabet._resolve_prostt5_device(
        torch_module=_fake_torch_module(cuda_available=False, mps_available=True),
        device_opt="mps",
    )
    assert dev == "mps"


def test_enable_mps_fallback_if_needed_sets_env_once(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    changed = structural_alphabet._enable_mps_fallback_if_needed(device="mps")
    assert changed is True
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    changed_again = structural_alphabet._enable_mps_fallback_if_needed(device="mps")
    assert changed_again is False


def test_enable_mps_fallback_if_needed_ignores_non_mps(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    changed = structural_alphabet._enable_mps_fallback_if_needed(device="cpu")
    assert changed is False
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ


def test_enable_mps_fallback_for_option_if_needed_auto_on_darwin(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(structural_alphabet.sys, "platform", "darwin")
    changed = structural_alphabet._enable_mps_fallback_for_option_if_needed(device_opt="auto")
    assert changed is True
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_enable_mps_fallback_for_option_if_needed_noop_off_darwin(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(structural_alphabet.sys, "platform", "linux")
    changed = structural_alphabet._enable_mps_fallback_for_option_if_needed(device_opt="auto")
    assert changed is False
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
