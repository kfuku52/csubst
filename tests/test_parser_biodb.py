import importlib
import sys
import types
import urllib
import numpy as np
import pytest
from csubst import tree
from csubst import ete


def _import_parser_biodb_with_fake_pymol(monkeypatch):
    fake_cmd = types.SimpleNamespace(
        get_fastastr=lambda **kwargs: "",
        get_chains=lambda *args, **kwargs: [],
        get_names=lambda *args, **kwargs: [],
        do=lambda *args, **kwargs: None,
    )
    fake_pymol = types.SimpleNamespace(cmd=fake_cmd)
    monkeypatch.setitem(sys.modules, "pymol", fake_pymol)
    sys.modules.pop("csubst.parser_pymol", None)
    sys.modules.pop("csubst.parser_biodb", None)
    return importlib.import_module("csubst.parser_biodb")


def test_get_top_hit_ids_parses_pipe_and_fallback_titles(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)

    class _Desc:
        def __init__(self, title):
            self.title = title

    class _Hits:
        descriptions = [
            _Desc("sp|P12345.7|SOME_PROTEIN"),
            _Desc("Q8XYZ1 hypothetical_protein"),
            _Desc(""),
        ]

    out = parser_biodb.get_top_hit_ids(_Hits())
    assert out == ["P12345", "Q8XYZ1"]


def test_run_qblast_returns_empty_list_when_no_descriptions(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)

    class _FakeSearch:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _Hits:
        descriptions = None

    fake_search = _FakeSearch()

    def _fake_qblast(**kwargs):
        return fake_search

    def _fake_read(search_obj):
        assert search_obj is fake_search
        return _Hits()

    monkeypatch.setattr(parser_biodb.NCBIWWW, "qblast", _fake_qblast)
    monkeypatch.setattr(parser_biodb.NCBIXML, "read", _fake_read)

    out = parser_biodb.run_qblast(aa_query="AAAA", num_display=1, evalue_cutoff=10)
    assert out == []
    assert fake_search.closed


def test_run_qblast_closes_handle_when_xml_parse_fails(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)

    class _FakeSearch:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_search = _FakeSearch()
    monkeypatch.setattr(parser_biodb.NCBIWWW, "qblast", lambda **kwargs: fake_search)

    def _raise_read(_search_obj):
        raise RuntimeError("xml parse failed")

    monkeypatch.setattr(parser_biodb.NCBIXML, "read", _raise_read)
    with pytest.raises(RuntimeError, match="xml parse failed"):
        parser_biodb.run_qblast(aa_query="AAAA", num_display=1, evalue_cutoff=10)
    assert fake_search.closed


def test_is_url_valid_returns_false_on_urlerror(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)

    def _bad_urlopen(_request, timeout=None):
        raise urllib.error.URLError("temporary network failure")

    monkeypatch.setattr(parser_biodb.urllib.request, "urlopen", _bad_urlopen)
    assert parser_biodb.is_url_valid("https://example.com/file.pdb") is False


def test_is_url_valid_closes_response_handle(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    state = {"closed": False}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            state["closed"] = True
            return False

    monkeypatch.setattr(parser_biodb.urllib.request, "urlopen", lambda _request, timeout=None: _Resp())
    assert parser_biodb.is_url_valid("https://example.com/file.pdb") is True
    assert state["closed"] is True


def test_is_url_valid_passes_timeout_to_urlopen(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    observed = {"timeout": None}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _urlopen(_request, timeout=None):
        observed["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(parser_biodb.urllib.request, "urlopen", _urlopen)
    assert parser_biodb.is_url_valid("https://example.com/file.pdb", timeout=7.5) is True
    assert observed["timeout"] == 7.5


def test_is_url_valid_falls_back_to_get_when_head_is_not_supported(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    calls = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _urlopen(request, timeout=None):
        method = request.get_method() if hasattr(request, "get_method") else "GET"
        calls.append(method)
        if method == "HEAD":
            raise urllib.error.HTTPError(
                url="https://example.com/file.pdb",
                code=405,
                msg="Method Not Allowed",
                hdrs=None,
                fp=None,
            )
        return _Resp()

    monkeypatch.setattr(parser_biodb.urllib.request, "urlopen", _urlopen)
    assert parser_biodb.is_url_valid("https://example.com/file.pdb") is True
    assert calls == ["HEAD", "GET"]


def test_is_url_valid_falls_back_to_get_when_head_is_forbidden(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    calls = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _urlopen(request, timeout=None):
        method = request.get_method() if hasattr(request, "get_method") else "GET"
        calls.append(method)
        if method == "HEAD":
            raise urllib.error.HTTPError(
                url="https://example.com/file.pdb",
                code=403,
                msg="Forbidden",
                hdrs=None,
                fp=None,
            )
        return _Resp()

    monkeypatch.setattr(parser_biodb.urllib.request, "urlopen", _urlopen)
    assert parser_biodb.is_url_valid("https://example.com/file.pdb") is True
    assert calls == ["HEAD", "GET"]


def test_is_url_valid_returns_false_when_head_unsupported_and_get_fails(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    calls = []

    def _urlopen(request, timeout=None):
        method = request.get_method() if hasattr(request, "get_method") else "GET"
        calls.append(method)
        if method == "HEAD":
            raise urllib.error.HTTPError(
                url="https://example.com/file.pdb",
                code=405,
                msg="Method Not Allowed",
                hdrs=None,
                fp=None,
            )
        raise urllib.error.URLError("GET failed")

    monkeypatch.setattr(parser_biodb.urllib.request, "urlopen", _urlopen)
    assert parser_biodb.is_url_valid("https://example.com/file.pdb") is False
    assert calls == ["HEAD", "GET"]


def test_get_representative_leaf_rejects_unknown_size(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    tr = parser_biodb.ete.PhyloNode("(A:1,B:1)R;", format=1)
    with pytest.raises(ValueError, match="Unsupported representative leaf size mode"):
        parser_biodb.get_representative_leaf(tr, size="shortest")


def test_pdb_sequence_search_rejects_unknown_representative_branch(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {"branch_ids": [999], "tree": tr}
    with pytest.raises(ValueError, match="Representative branch ID 999 was not found"):
        parser_biodb.pdb_sequence_search(g)


def test_pdb_sequence_search_rejects_none_branch_ids_with_clear_error(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {"branch_ids": None, "tree": tr}
    with pytest.raises(ValueError, match="No branch IDs were provided"):
        parser_biodb.pdb_sequence_search(g)


def test_normalize_branch_ids_rejects_non_integer_like_values(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    with pytest.raises(ValueError, match="integer-like"):
        parser_biodb._normalize_branch_ids([1.5])
    with pytest.raises(ValueError, match="integer-like"):
        parser_biodb._normalize_branch_ids([True])


def test_pdb_sequence_search_accepts_scalar_branch_id(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    g = {"branch_ids": np.int64(999), "tree": tr}
    with pytest.raises(ValueError, match="Representative branch ID 999 was not found"):
        parser_biodb.pdb_sequence_search(g)


def test_pdb_sequence_search_alphafill_handles_qblast_failure(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    a_bid = int(ete.get_prop(a_node, "numerical_label"))
    monkeypatch.setattr(parser_biodb.sequence, "translate_state", lambda nlabel, mode, g: "AAAA")
    monkeypatch.setattr(
        parser_biodb,
        "run_qblast",
        lambda aa_query, num_display=10, evalue_cutoff=10: (_ for _ in ()).throw(RuntimeError("qblast failed")),
    )
    g = {
        "branch_ids": [a_bid],
        "tree": tr,
        "database": "alphafill",
        "database_evalue_cutoff": 10,
        "database_minimum_identity": 0.0,
        "pymol_max_num_chain": 999,
    }
    out = parser_biodb.pdb_sequence_search(g)
    assert out["pdb"] is None


def test_pdb_sequence_search_strips_database_tokens(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    a_bid = int(ete.get_prop(a_node, "numerical_label"))
    monkeypatch.setattr(parser_biodb.sequence, "translate_state", lambda nlabel, mode, g: "AAAA")
    monkeypatch.setattr(
        parser_biodb.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network disabled")),
    )
    call_count = {"run_qblast": 0}

    def _fake_run_qblast(aa_query, num_display=10, evalue_cutoff=10):
        call_count["run_qblast"] += 1
        return []

    monkeypatch.setattr(parser_biodb, "run_qblast", _fake_run_qblast)
    g = {
        "branch_ids": [a_bid],
        "tree": tr,
        "database": "pdb, alphafill",
        "database_evalue_cutoff": 10,
        "database_minimum_identity": 0.0,
        "pymol_max_num_chain": 999,
    }
    out = parser_biodb.pdb_sequence_search(g)
    assert out["pdb"] is None
    assert call_count["run_qblast"] == 1


def test_pdb_sequence_search_pdb_request_uses_timeout_and_raise_for_status(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    a_bid = int(ete.get_prop(a_node, "numerical_label"))
    monkeypatch.setattr(parser_biodb.sequence, "translate_state", lambda nlabel, mode, g: "AAAA")
    state = {"timeout": None, "raise_called": False}

    class _Response:
        def raise_for_status(self):
            state["raise_called"] = True

        def json(self):
            return {"result_set": []}

    def _fake_post(*args, **kwargs):
        state["timeout"] = kwargs.get("timeout")
        return _Response()

    monkeypatch.setattr(parser_biodb.requests, "post", _fake_post)
    g = {
        "branch_ids": [a_bid],
        "tree": tr,
        "database": "pdb",
        "database_timeout": 12,
        "database_evalue_cutoff": 10,
        "database_minimum_identity": 0.0,
        "pymol_max_num_chain": 999,
    }
    out = parser_biodb.pdb_sequence_search(g)
    assert out["pdb"] is None
    assert state["timeout"] == 12.0
    assert state["raise_called"] is True


def test_pdb_sequence_search_rejects_unknown_database_names(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    a_bid = int(ete.get_prop(a_node, "numerical_label"))
    monkeypatch.setattr(parser_biodb.sequence, "translate_state", lambda nlabel, mode, g: "AAAA")
    g = {
        "branch_ids": [a_bid],
        "tree": tr,
        "database": "pdb,unknown_db",
        "database_evalue_cutoff": 10,
        "database_minimum_identity": 0.0,
        "pymol_max_num_chain": 999,
    }
    with pytest.raises(ValueError, match="Unknown database name"):
        parser_biodb.pdb_sequence_search(g)


def test_pdb_sequence_search_rejects_empty_database_list(monkeypatch):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    a_bid = int(ete.get_prop(a_node, "numerical_label"))
    monkeypatch.setattr(parser_biodb.sequence, "translate_state", lambda nlabel, mode, g: "AAAA")
    g = {
        "branch_ids": [a_bid],
        "tree": tr,
        "database": " , ",
        "database_evalue_cutoff": 10,
        "database_minimum_identity": 0.0,
        "pymol_max_num_chain": 999,
    }
    with pytest.raises(ValueError, match="No database was specified"):
        parser_biodb.pdb_sequence_search(g)


def test_pdb_sequence_search_alphafold_download_uses_context_manager(monkeypatch, tmp_path):
    parser_biodb = _import_parser_biodb_with_fake_pymol(monkeypatch)
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    a_node = [n for n in tr.traverse() if n.name == "A"][0]
    a_bid = int(ete.get_prop(a_node, "numerical_label"))
    monkeypatch.setattr(parser_biodb.sequence, "translate_state", lambda nlabel, mode, g: "AAAA")
    monkeypatch.setattr(parser_biodb, "run_qblast", lambda aa_query, num_display=10, evalue_cutoff=10: ["P12345"])
    monkeypatch.setattr(parser_biodb, "is_url_valid", lambda url, timeout=30: True)
    monkeypatch.chdir(tmp_path)
    state = {"closed": False, "timeout": None}

    class _Inner:
        def read(self):
            return b"MODEL"

    class _UrlOpenResult:
        def __enter__(self):
            return _Inner()

        def __exit__(self, exc_type, exc, tb):
            state["closed"] = True
            return False

    def _urlopen(_url, timeout=None):
        state["timeout"] = timeout
        return _UrlOpenResult()

    monkeypatch.setattr(parser_biodb.urllib.request, "urlopen", _urlopen)
    g = {
        "branch_ids": [a_bid],
        "tree": tr,
        "database": "alphafold",
        "database_timeout": 9,
        "database_evalue_cutoff": 10,
        "database_minimum_identity": 0.0,
        "pymol_max_num_chain": 999,
    }
    out = parser_biodb.pdb_sequence_search(g)
    assert out["selected_database"] == "alphafold"
    assert out["pdb"] == "AF-P12345-F1-model_v2.pdb"
    assert state["closed"] is True
    assert state["timeout"] == 9.0
