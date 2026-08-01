import numpy as np
import pytest

from csubst import tree
from csubst import ete


class _FakeTreeAxis:
    def __init__(self):
        self.plot_calls = []
        self.scatter_calls = []
        self.text_calls = []
        self.text_items = []
        self.patch_calls = []
        self.collection_calls = []
        self.xlim_calls = []
        self.legend_calls = []
        self.transAxes = object()
        self.transData = object()
        self.figure = None

    def plot(self, *args, **kwargs):
        self.plot_calls.append((args, kwargs))
        return None

    def scatter(self, *args, **kwargs):
        self.scatter_calls.append((args, kwargs))
        return None

    def text(self, x, y, txt, **kwargs):
        self.text_calls.append(str(txt))
        self.text_items.append({
            "x": x,
            "y": y,
            "txt": str(txt),
            "kwargs": dict(kwargs),
        })
        return None

    def add_patch(self, patch):
        self.patch_calls.append(patch)
        return patch

    def add_collection(self, collection):
        self.collection_calls.append(collection)
        return collection

    def set_xlim(self, *args, **kwargs):
        self.xlim_calls.append((args, kwargs))
        return None

    def set_ylim(self, *args, **kwargs):
        return None

    def axis(self, *args, **kwargs):
        return None

    def get_position(self):
        return type("FakeAxisPosition", (), {"width": 1.0})()

    def legend(self, *args, **kwargs):
        self.legend_calls.append((args, kwargs))
        return None


class _FakeTreeFigure:
    def __init__(self):
        self.savefig_calls = []
        self.subplots_adjust_calls = []
        self.suptitle_calls = []
        self.text_calls = []
        self.text_items = []
        self.transFigure = object()

    def savefig(self, *args, **kwargs):
        self.savefig_calls.append((args, kwargs))
        return None

    def suptitle(self, *args, **kwargs):
        self.suptitle_calls.append((args, kwargs))
        return None

    def subplots_adjust(self, *args, **kwargs):
        self.subplots_adjust_calls.append((args, kwargs))
        return None

    def text(self, x, y, txt, **kwargs):
        self.text_calls.append(str(txt))
        self.text_items.append({
            "x": x,
            "y": y,
            "txt": str(txt),
            "kwargs": dict(kwargs),
        })
        return None

    def get_size_inches(self):
        return (tree.TREE_FIG_WIDTH, 4.0)


class _FakeTreePyplot:
    def __init__(self):
        self.axis = _FakeTreeAxis()
        self.figure = _FakeTreeFigure()
        self.axis.figure = self.figure

    def subplots(self, *args, **kwargs):
        return self.figure, self.axis

    def close(self, *args, **kwargs):
        return None


def test_render_tree_matplotlib_hides_missing_root_state_for_codon(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    fake_plt = _FakeTreePyplot()
    monkeypatch.setattr(tree, "_get_pyplot", lambda: fake_plt)

    tree._render_tree_matplotlib(
        tree=tr,
        trait_name="trait",
        file_name=str(tmp_path / "state_codon.pdf"),
        state_by_node={
            labels["R"]: "---",
            labels["A"]: "AAA",
            labels["B"]: "AAG",
        },
        state_mode="codon",
    )

    assert "---" not in fake_plt.axis.text_calls
    assert "AAA|A" in fake_plt.axis.text_calls
    assert "AAG|B" in fake_plt.axis.text_calls
    assert any(txt.endswith("subs/codon site") for txt in fake_plt.axis.text_calls)
    assert fake_plt.figure.subplots_adjust_calls[-1][1] == {
        "left": 0.0,
        "right": 1.0,
        "bottom": 0.0,
        "top": 1.0,
    }
    assert fake_plt.figure.savefig_calls[-1][1]["pad_inches"] == tree.TREE_FIG_SAVE_PAD_INCHES
    scale_label_item = next(item for item in fake_plt.axis.text_items if item["txt"].endswith("subs/codon site"))
    expected_scale_label_y = tree.TREE_SCALE_BAR_Y + tree.TREE_SCALE_BAR_TICK_HALF_HEIGHT + tree.TREE_SCALE_BAR_LABEL_GAP
    assert pytest.approx(scale_label_item["y"], rel=0, abs=1e-12) == expected_scale_label_y
    assert scale_label_item["kwargs"]["va"] == "bottom"


def test_render_tree_matplotlib_places_branch_ids_below_branch_midpoints(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:2,B:4)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    xcoord,ycoord,_ = tree._get_tree_xy(tr)
    expected_gap = max(
        tree._get_text_height_in_data_units(
            ax=_FakeTreeAxis(),
            fontsize=tree.TREE_BRANCH_ID_TEXT_SIZE,
            fallback_height=tree.AA_LOGO_HEIGHT_FALLBACK * 0.5,
        ) * 0.2,
        tree.TREE_BRANCH_ID_MIN_GAP,
    )
    fake_plt = _FakeTreePyplot()
    monkeypatch.setattr(tree, "_get_pyplot", lambda: fake_plt)

    tree._render_tree_matplotlib(
        tree=tr,
        trait_name="trait",
        file_name=str(tmp_path / "branch_id.pdf"),
        label="all",
    )

    items_by_text = {item["txt"]: item for item in fake_plt.axis.text_items}
    for node in tr.traverse():
        if ete.is_root(node):
            continue
        branch_id_text = tree._format_branch_id_label(ete.get_prop(node, "numerical_label"))
        text_item = items_by_text[branch_id_text]
        expected_x = (xcoord[id(node.up)] + xcoord[id(node)]) / 2.0
        expected_y = ycoord[id(node)] - expected_gap
        assert pytest.approx(text_item["x"], rel=0, abs=1e-12) == expected_x
        assert pytest.approx(text_item["y"], rel=0, abs=1e-12) == expected_y
        assert text_item["kwargs"]["fontsize"] == tree.TREE_BRANCH_ID_TEXT_SIZE
        assert text_item["kwargs"]["va"] == "top"
        assert text_item["kwargs"]["ha"] == "center"


def test_resolve_species_overlap_node_types_auto_detects_duplication_and_speciation():
    tr = tree.add_numerical_node_labels(ete.PhyloNode(
        "((Homo_sapiens_gene1:1,Homo_sapiens_gene2:1)Dup:1,(Mus_musculus_gene1:1,Rattus_norvegicus_gene1:1)Spec:1)Root;",
        format=1,
    ))
    labels = {node.name: int(ete.get_prop(node, "numerical_label")) for node in tr.traverse()}
    node_types = tree._resolve_species_overlap_node_types(
        tree=tr,
        species_regex='^([^_]+_[^_]+)_',
        species_overlap_node_plot='auto',
    )
    assert node_types[labels["Dup"]] == "duplication"
    assert node_types[labels["Spec"]] == "speciation"


def test_render_tree_matplotlib_draws_species_overlap_node_markers(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    root_id = int(ete.get_prop(tr, "numerical_label"))
    fake_plt = _FakeTreePyplot()
    monkeypatch.setattr(tree, "_get_pyplot", lambda: fake_plt)

    tree._render_tree_matplotlib(
        tree=tr,
        trait_name="trait",
        file_name=str(tmp_path / "marker.pdf"),
        state_by_node=None,
        node_type_by_id={root_id: "duplication"},
    )

    assert len(fake_plt.axis.scatter_calls) == 1
    scatter_args, scatter_kwargs = fake_plt.axis.scatter_calls[0]
    assert scatter_kwargs["facecolor"] == tree.TREE_DUPLICATION_COLOR
    assert scatter_kwargs["s"] == tree.TREE_NODE_MARKER_AREA
    assert scatter_kwargs["zorder"] == tree.TREE_NODE_MARKER_ZORDER
    assert len(fake_plt.axis.legend_calls) == 1
    _, legend_kwargs = fake_plt.axis.legend_calls[0]
    assert legend_kwargs["loc"] == "upper left"
    assert legend_kwargs["bbox_to_anchor"] == (tree.TREE_NODE_LEGEND_X, tree.TREE_NODE_LEGEND_Y)
    assert [handle.get_label() for handle in legend_kwargs["handles"]] == [
        "Speciation node",
        "Duplication node",
    ]


def test_render_tree_matplotlib_uses_projecting_caps_for_internal_segments(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    fake_plt = _FakeTreePyplot()
    monkeypatch.setattr(tree, "_get_pyplot", lambda: fake_plt)

    tree._render_tree_matplotlib(
        tree=tr,
        trait_name="trait",
        file_name=str(tmp_path / "caps.pdf"),
        label="all",
    )

    capstyles = [collection.get_capstyle() for collection in fake_plt.axis.collection_calls]
    assert tree.TREE_LINE_CAPSTYLE in capstyles
    assert tree.TREE_LINE_TERMINAL_CAPSTYLE in capstyles
    assert all(collection.get_joinstyle() == tree.TREE_LINE_JOINSTYLE for collection in fake_plt.axis.collection_calls)


def test_render_tree_matplotlib_hides_missing_root_state_for_aa(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    fake_plt = _FakeTreePyplot()
    monkeypatch.setattr(tree, "_get_pyplot", lambda: fake_plt)
    monkeypatch.setattr(tree, "_get_logo_modules", lambda: (None, None, None, None))

    tree._render_tree_matplotlib(
        tree=tr,
        trait_name="trait",
        file_name=str(tmp_path / "state_aa.pdf"),
        state_by_node={
            labels["R"]: "-",
            labels["A"]: "K",
            labels["B"]: "N",
        },
        state_prob_by_node={
            labels["R"]: None,
            labels["A"]: None,
            labels["B"]: None,
        },
        state_orders=np.array(["K", "N"], dtype=object),
        state_mode="aa",
    )

    assert "-" not in fake_plt.axis.text_calls
    assert "K" in fake_plt.axis.text_calls
    assert "N" in fake_plt.axis.text_calls


def test_render_tree_matplotlib_offsets_root_state_text_to_clear_root_marker(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    fake_plt = _FakeTreePyplot()
    monkeypatch.setattr(tree, "_get_pyplot", lambda: fake_plt)

    tree._render_tree_matplotlib(
        tree=tr,
        trait_name="trait",
        file_name=str(tmp_path / "state_root_offset.pdf"),
        state_by_node={
            labels["R"]: "AAA",
            labels["A"]: "AAC",
            labels["B"]: "AAG",
        },
        state_mode="codon",
        node_type_by_id={labels["R"]: "duplication"},
    )

    xcoord,_,_ = tree._get_tree_xy(tr)
    xspan = max(xcoord.values())
    root_item = next(item for item in fake_plt.axis.text_items if item["txt"] == "AAA")
    expected_x = xcoord[id(tr)] + (xspan * (tree.TREE_STATE_X_PADDING_RATIO + tree.TREE_ROOT_STATE_EXTRA_X_PADDING_RATIO))
    assert pytest.approx(root_item["x"], rel=0, abs=1e-12) == expected_x


def test_render_tree_matplotlib_places_figure_title_in_figure_coordinates(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    fake_plt = _FakeTreePyplot()
    monkeypatch.setattr(tree, "_get_pyplot", lambda: fake_plt)

    tree._render_tree_matplotlib(
        tree=tr,
        trait_name="trait",
        file_name=str(tmp_path / "state_title.pdf"),
        figure_title="Sites 1-2-3",
    )

    title_item = [item for item in fake_plt.figure.text_items if item["txt"] == "Sites 1-2-3"][-1]
    assert title_item["x"] == tree.TREE_FIG_TITLE_X
    assert title_item["y"] == tree.TREE_FIG_TITLE_Y
    assert title_item["kwargs"]["ha"] == "left"
    assert title_item["kwargs"]["va"] == "top"
    assert "Sites 1-2-3" not in fake_plt.axis.text_calls
    assert fake_plt.figure.suptitle_calls == []


def test_render_tree_matplotlib_draws_placeholder_for_missing_aa_logo_sites(monkeypatch, tmp_path):
    tr = tree.add_numerical_node_labels(ete.PhyloNode("(A:1,B:1)R;", format=1))
    for node in tr.traverse():
        ete.set_prop(node, "color_trait", "black")
        ete.set_prop(node, "labelcolor_trait", "black")
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse()}
    fake_plt = _FakeTreePyplot()
    monkeypatch.setattr(tree, "_get_pyplot", lambda: fake_plt)
    monkeypatch.setattr(tree, "_get_logo_modules", lambda: (None, None, None, None))

    def fake_draw_logo(ax, x, y, probabilities, orders, logo_width, logo_height,
                       mpl_patches, mpl_textpath, mpl_transforms, font_properties):
        return bool(np.asarray(probabilities, dtype=float).sum() > 0)

    monkeypatch.setattr(tree, "_draw_aa_logo", fake_draw_logo)

    tree._render_tree_matplotlib(
        tree=tr,
        trait_name="trait",
        file_name=str(tmp_path / "state_missing_logo.pdf"),
        state_by_node={
            labels["R"]: "----",
            labels["A"]: "A-CC",
            labels["B"]: "CCCC",
        },
        state_prob_by_node={
            labels["R"]: None,
            labels["A"]: np.array([
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ], dtype=float),
            labels["B"]: np.ones((4, 2), dtype=float),
        },
        state_orders=np.array(["A", "C"], dtype=object),
        state_mode="aa",
    )

    assert len(fake_plt.axis.patch_calls) == 2
    assert all(type(patch).__name__ == "Rectangle" for patch in fake_plt.axis.patch_calls)
    assert "A" in fake_plt.axis.text_calls
    assert "B" in fake_plt.axis.text_calls
