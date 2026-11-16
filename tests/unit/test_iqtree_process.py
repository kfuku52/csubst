
import numpy as np
import pytest

from csubst import parser_iqtree
from csubst import tree
from csubst import ete


def test_mask_missing_sites_nonbinary_internal_uses_all_child_groups():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1,C:1)N1:1,D:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    state = np.zeros((len(list(tr.traverse())), 1, 1), dtype=float)
    # Only one child clade (C) and one sister clade (D) have data at this site.
    state[labels["C"], 0, 0] = 1.0
    state[labels["D"], 0, 0] = 1.0
    state[labels["N1"], 0, 0] = 1.0
    out = parser_iqtree.mask_missing_sites(state_tensor=state, tree=tr)
    assert out[labels["N1"], 0, 0] == pytest.approx(1.0)


def test_run_iqtree_ancestral_nonzero_exit_raises_clear_error_and_cleans_tmp_tree(tmp_path, monkeypatch):
    alignment_file = tmp_path / "toy.fa"
    alignment_file.write_text(">A\nAAA\n>B\nAAA\n", encoding="utf-8")
    rooted_tree = ete.PhyloNode("(A:1,B:1)R;", format=1)
    g = {
        "rooted_tree": rooted_tree,
        "alignment_file": str(alignment_file),
        "iqtree_exe": "iqtree2",
        "iqtree_model": "GY",
        "genetic_code": 1,
        "threads": 1,
    }

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(parser_iqtree.tree, "is_consistent_tree_and_aln", lambda g: True)

    def fake_write_tree(tree_obj, outfile, add_numerical_label=False):
        (tmp_path / outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    monkeypatch.setattr(parser_iqtree.tree, "write_tree", fake_write_tree)
    monkeypatch.setattr(parser_iqtree.runtime, "run_subprocess_tee", lambda command: 2)

    with pytest.raises(AssertionError, match="exit code 2"):
        parser_iqtree.run_iqtree_ancestral(g)
    assert not (tmp_path / "tmp.csubst.nwk").exists()


def test_run_iqtree_ancestral_rejects_inconsistent_tree_without_force(tmp_path, monkeypatch):
    alignment_file = tmp_path / "toy.fa"
    alignment_file.write_text(">A\nAAA\n>B\nAAA\n", encoding="utf-8")
    rooted_tree = ete.PhyloNode("(A:1,B:1)R;", format=1)
    g = {
        "rooted_tree": rooted_tree,
        "alignment_file": str(alignment_file),
        "iqtree_exe": "iqtree2",
        "iqtree_model": "GY",
        "genetic_code": 1,
        "threads": 1,
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(parser_iqtree.tree, "is_consistent_tree_and_aln", lambda g: False)

    def fake_write_tree(tree_obj, outfile, add_numerical_label=False):
        (tmp_path / outfile).write_text("(A:1,B:1)R;\n", encoding="utf-8")

    monkeypatch.setattr(parser_iqtree.tree, "write_tree", fake_write_tree)
    with pytest.raises(ValueError, match="not consistent"):
        parser_iqtree.run_iqtree_ancestral(g, force_notree_run=False)
    assert not (tmp_path / "tmp.csubst.nwk").exists()


@pytest.mark.parametrize('calibration', ['none', 'parametric_bootstrap'])
@pytest.mark.parametrize('observation', ['marginal', 'joint', 'bridge'])
def test_scan_fit_passes_seed_and_retains_checkpoint_for_precise_models(tmp_path, monkeypatch, calibration, observation):
    from pathlib import Path
    alignment = tmp_path/'input.fa'
    alignment.write_text('>A\nAAA\n>B\nAAC\n')
    g = dict(rooted_tree=ete.PhyloNode('(A:.1,B:.2)R;', format=1), alignment_file=str(alignment),
             iqtree_exe='iqtree2', iqtree_model='GY+F', genetic_code=1, threads=1,
             subcommand='scan', random_seed=5105, scan_pvalue_calibration=calibration, scan_observation=observation,
             iqtree_outdir=str(tmp_path/'iqtree'))
    created = []
    def fake_fit(command):
        assert command[command.index('-seed')+1] == '5105'
        assert '-v' in command
        checkpoint = Path(command[command.index('--prefix')+1] + '.ckp.gz')
        checkpoint.write_bytes(b'checkpoint fixture')
        created.append(checkpoint)
        return 0
    monkeypatch.setattr(parser_iqtree.runtime, 'run_subprocess_tee', fake_fit)
    monkeypatch.setattr(parser_iqtree, '_write_iqtree_manifest', lambda g: None)
    parser_iqtree.run_iqtree_ancestral(g)
    assert created[0].exists()  # GY sidecars are shared by every estimator and command.


@pytest.mark.parametrize('retry_gain', [None, 0., 10.])
def test_guarded_fit_publishes_only_a_verified_improvement(tmp_path, monkeypatch, retry_gain):
    from pathlib import Path

    from csubst import iqtree_fit

    extensions = (*parser_iqtree._FIT_EXTENSIONS, 'ckp.gz')
    prefix = str(tmp_path / 'fit')
    paths = [Path(prefix + '.' + ext) for ext in extensions]
    manifest = Path(prefix + '.state.csubst-manifest.json')
    for path in (*paths, manifest):
        path.write_text('existing')
    initial = iqtree_fit.OmegaCheck(1, -100, -100, 5 if retry_gain is not None else -.1, .002)
    calls = []

    def fake_fit(command):
        calls.append(command)
        staged = command[command.index('--prefix') + 1]
        assert staged != prefix
        for ext in extensions:
            Path(staged + '.' + ext).write_text('retry' if '--inc-zero-freq' in command else 'initial')
        return 0

    def fake_audit(g, verify_refit=False):
        assert Path(g['path_iqtree_state']).read_text() == ('retry' if verify_refit else 'initial')
        if verify_refit:
            assert retry_gain is not None
            return iqtree_fit.OmegaCheck(.4, -100 + retry_gain, -100 + retry_gain, -.1, .002)
        return initial

    monkeypatch.setattr(parser_iqtree.runtime, 'run_subprocess_tee', fake_fit)
    monkeypatch.setattr(iqtree_fit, 'audit_hybrid_omega', fake_audit)
    monkeypatch.setattr(parser_iqtree, '_write_iqtree_manifest', lambda g: manifest.write_text('accepted'))
    if retry_gain is None or retry_gain > 0:
        parser_iqtree._run_guarded_hybrid({}, ['iqtree', '--prefix', prefix], prefix, None)
        assert all(path.read_text() == ('initial' if retry_gain is None else 'retry') for path in paths)
        assert manifest.read_text() == 'accepted'
    else:
        with pytest.raises(ValueError, match='optimization did not recover'):
            parser_iqtree._run_guarded_hybrid({}, ['iqtree', '--prefix', prefix], prefix, None)
        assert all(path.read_text() == 'existing' for path in (*paths, manifest))
    assert len(calls) == (1 if retry_gain is None else 2)
    if retry_gain is not None:
        assert calls[1][-3:] == ['--inc-zero-freq', '--min-freq', '0.0001']
    assert not list(tmp_path.glob('.csubst-iqtree-*'))
