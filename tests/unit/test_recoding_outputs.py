import numpy as np
import pytest
from matplotlib import image as mpimg

from csubst import recoding


def _toy_grouping_g():
    amino_acids = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object)
    codon_orders = np.array(["C{:02d}".format(i) for i in range(amino_acids.shape[0])], dtype=object)
    synonymous_indices = {aa: [i] for i, aa in enumerate(amino_acids.tolist())}
    matrix_groups = {aa: [codon_orders[i]] for i, aa in enumerate(amino_acids.tolist())}
    return {
        "amino_acid_orders": amino_acids,
        "codon_orders": codon_orders,
        "synonymous_indices": synonymous_indices,
        "matrix_groups": matrix_groups,
    }


def _toy_auto_grouping_g():
    g = _toy_grouping_g()
    aa_orders = [str(aa) for aa in g["amino_acid_orders"].tolist()]
    aa_matrix = np.vstack(
        [
            np.arange(20, dtype=np.int16),
            np.roll(np.arange(20, dtype=np.int16), 1),
            np.roll(np.arange(20, dtype=np.int16), 5),
            np.roll(np.arange(20, dtype=np.int16), 10),
        ]
    )
    base = np.arange(1, 21, dtype=np.float64)
    fmat = np.vstack(
        [
            base / base.sum(),
            base[::-1] / base.sum(),
            np.roll(base, 5) / base.sum(),
            np.roll(base, 10) / base.sum(),
        ]
    )
    nsitev = np.array([400, 400, 400, 400], dtype=np.int64)
    fr = (fmat * nsitev[:, np.newaxis]).sum(axis=0)
    fr = fr / fr.sum()
    g["alignment_file"] = ""
    g["nonsyn_recode_seed"] = 7
    g["nonsyn_recode_random_starts"] = 24
    g["_nonsyn_recode_alignment_cache"] = {
        "alignment_file": "",
        "aa_orders": tuple(aa_orders),
        "aa_matrix": aa_matrix,
        "fmat": fmat,
        "fr": fr,
        "nsitev": nsitev,
    }
    return g


def test_write_nonsyn_recoding_table_writes_non_none_scheme(tmp_path):
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    g = recoding.initialize_nonsyn_groups(g)
    output_path = tmp_path / "csubst_nonsyn_recoding.tsv"
    returned = recoding.write_nonsyn_recoding_table(g, output_path=str(output_path))
    assert returned == str(output_path)
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("recode\tstate_id\tstate_label")
    assert len(lines) == 1 + len(g["amino_acid_orders"])
    assert any([line.split("\t")[4] == "A" for line in lines[1:]])


def test_write_nonsyn_recoding_table_skips_no(tmp_path):
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "no"
    g = recoding.initialize_nonsyn_groups(g)
    output_path = tmp_path / "csubst_nonsyn_recoding.tsv"
    returned = recoding.write_nonsyn_recoding_table(g, output_path=str(output_path))
    assert returned is None
    assert output_path.exists() is False


def test_write_nonsyn_recoding_pca_plot_writes_png_for_fixed_recode(tmp_path):
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    g = recoding.initialize_nonsyn_groups(g)
    output_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    returned = recoding.write_nonsyn_recoding_pca_plot(g, output_path=str(output_path))
    assert returned == str(output_path)
    assert output_path.exists() is True
    assert output_path.stat().st_size > 0
    img = mpimg.imread(str(output_path))
    assert img.shape[0] == 720
    assert img.shape[1] == 720


def test_write_nonsyn_recoding_pca_plot_writes_png_for_auto_recode(tmp_path):
    g = _toy_auto_grouping_g()
    g["nonsyn_recode"] = "srchisq6"
    g = recoding.initialize_nonsyn_groups(g)
    output_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    returned = recoding.write_nonsyn_recoding_pca_plot(g, output_path=str(output_path))
    assert returned == str(output_path)
    assert output_path.exists() is True
    assert output_path.stat().st_size > 0


def test_get_label_connector_mask_marks_far_labels():
    point_x = np.array([0.0, 0.0], dtype=np.float64)
    point_y = np.array([0.0, 0.0], dtype=np.float64)
    label_x = np.array([0.012, 0.012], dtype=np.float64)
    label_y = np.array([0.0, 0.09], dtype=np.float64)
    out = recoding._get_label_connector_mask(
        point_x=point_x,
        point_y=point_y,
        label_x=label_x,
        label_y=label_y,
        x_span=1.0,
        y_span=1.0,
        normalized_threshold=0.03,
    )
    assert out.dtype == bool
    assert out.tolist() == [False, True]


def test_get_label_connector_mask_raises_on_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        recoding._get_label_connector_mask(
            point_x=np.array([0.0, 1.0], dtype=np.float64),
            point_y=np.array([0.0], dtype=np.float64),
            label_x=np.array([0.0, 1.0], dtype=np.float64),
            label_y=np.array([0.0, 1.0], dtype=np.float64),
            x_span=1.0,
            y_span=1.0,
        )


def test_is_outside_axis_outlier_detects_large_outside_shift():
    other = np.array([-1.0, -0.2, 0.1, 0.7, 1.0], dtype=np.float64)
    assert recoding._is_outside_axis_outlier(value=3.0, other_values=other) is True
    assert recoding._is_outside_axis_outlier(value=0.5, other_values=other) is False


def test_detect_srchisq6_inset_target_true_for_far_point():
    names = ["no", "dayhoff6", "sr6", "srchisq6", "3di20"]
    x = np.array([0.0, 0.1, -0.1, 3.0, 0.2], dtype=np.float64)
    y = np.array([0.0, 0.2, -0.2, 4.0, 0.1], dtype=np.float64)
    out = recoding._detect_srchisq6_inset_target(scheme_names=names, x=x, y=y)
    assert out["show_inset"] is True
    assert out["srchisq6_index"] == 3
    assert out["x_far"] is True
    assert out["y_far"] is True


def test_detect_srchisq6_inset_target_false_for_compact_point_cloud():
    names = ["no", "dayhoff6", "sr6", "srchisq6", "3di20"]
    x = np.array([0.0, 0.1, -0.1, 0.08, 0.2], dtype=np.float64)
    y = np.array([0.0, 0.2, -0.2, 0.12, 0.1], dtype=np.float64)
    out = recoding._detect_srchisq6_inset_target(scheme_names=names, x=x, y=y)
    assert out["show_inset"] is False
    assert out["srchisq6_index"] == 3
    assert out["x_far"] is False
    assert out["y_far"] is False


def test_get_scheme_groups_for_pca_includes_auto_schemes_when_data_available():
    g = _toy_auto_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    g = recoding.initialize_nonsyn_groups(g)
    groups_by_scheme = recoding._get_scheme_groups_for_pca(g)
    assert "no" in groups_by_scheme
    assert groups_by_scheme["no"] == tuple(list("ACDEFGHIKLMNPQRSTVWY"))
    assert "srchisq6" in groups_by_scheme
    assert "kgbauto6" in groups_by_scheme
    assert len(groups_by_scheme["srchisq6"]) == 6
    assert len(groups_by_scheme["kgbauto6"]) == 6


def test_get_scheme_groups_for_pca_3di20_is_optional():
    g = _toy_auto_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    g = recoding.initialize_nonsyn_groups(g)
    groups_default = recoding._get_scheme_groups_for_pca(g)
    groups_with_3di = recoding._get_scheme_groups_for_pca(g, include_3di20=True)
    assert "3di20" not in groups_default
    assert "3di20" in groups_with_3di
    assert groups_with_3di["3di20"] == tuple(list("ACDEFGHIKLMNPQRSTVWY"))


def test_get_scheme_groups_for_pca_require_auto_raises_when_missing():
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    g["alignment_file"] = "/tmp/nonexistent_alignment.fa"
    g = recoding.initialize_nonsyn_groups(g)
    with pytest.raises(ValueError, match="Failed to infer auto recoding scheme"):
        recoding._get_scheme_groups_for_pca(g, require_auto=True)


def test_build_3di20_dataset_feature_vector_is_not_no_when_data_available(monkeypatch):
    from csubst import structural_alphabet

    g = _toy_grouping_g()
    g["tree"] = object()
    g["codon_table"] = [("A", "GCT")]

    aa_by_tip = {
        "tip1": "AAAA",
        "tip2": "CCCC",
        "tip3": "AACC",
    }
    threedi_by_tip = {
        "tip1": "ABAB",
        "tip2": "CDCD",
        "tip3": "ABCD",
    }

    monkeypatch.setattr(
        structural_alphabet,
        "build_tip_aa_alignment_from_full_cds",
        lambda g: aa_by_tip,
    )
    monkeypatch.setattr(
        structural_alphabet,
        "build_tip_3di_alignment_from_full_cds",
        lambda g, output_path=None: threedi_by_tip,
    )
    monkeypatch.setattr(
        structural_alphabet,
        "get_3di_state_orders",
        lambda: np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype=object),
    )

    v_no = recoding._build_co_cluster_feature_vector(tuple(list("ACDEFGHIKLMNPQRSTVWY")))
    v_3di = recoding._build_3di20_dataset_feature_vector(g=g)
    assert v_3di is not None
    assert v_3di.shape == v_no.shape
    assert np.all(np.isfinite(v_3di))
    assert np.allclose(v_3di, v_no) is False


def test_write_nonsyn_recoding_pca_plot_writes_png_for_no(tmp_path):
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "no"
    g = recoding.initialize_nonsyn_groups(g)
    output_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    returned = recoding.write_nonsyn_recoding_pca_plot(g, output_path=str(output_path))
    assert returned == str(output_path)
    assert output_path.exists() is True
    assert output_path.stat().st_size > 0


def test_write_nonsyn_recoding_pca_plot_skips_3di20_when_disabled(tmp_path, monkeypatch):
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    g["plot_nonsyn_recode_pca_3di20"] = False
    g = recoding.initialize_nonsyn_groups(g)
    monkeypatch.setattr(
        recoding,
        "_build_3di20_dataset_feature_vector",
        lambda g: (_ for _ in ()).throw(AssertionError("3di20 feature should be skipped")),
    )
    output_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    returned = recoding.write_nonsyn_recoding_pca_plot(g, output_path=str(output_path))
    assert returned == str(output_path)
    assert output_path.exists() is True


def test_write_nonsyn_recoding_pca_plot_includes_3di20_when_enabled(tmp_path, monkeypatch):
    g = _toy_grouping_g()
    g["nonsyn_recode"] = "dayhoff6"
    g["plot_nonsyn_recode_pca_3di20"] = True
    g = recoding.initialize_nonsyn_groups(g)
    called = {"used": False}

    def _fake_build_3di20(g):
        called["used"] = True
        return np.linspace(0.0, 1.0, num=190, dtype=np.float64)

    monkeypatch.setattr(recoding, "_build_3di20_dataset_feature_vector", _fake_build_3di20)
    output_path = tmp_path / "csubst_nonsyn_recoding_pca.png"
    returned = recoding.write_nonsyn_recoding_pca_plot(g, output_path=str(output_path))
    assert returned == str(output_path)
    assert output_path.exists() is True
    assert called["used"] is True
