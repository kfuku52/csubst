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


def test_draw_logo_placeholder_uses_centered_inset_slot_dimensions():
    axis = _FakeTreeAxis()

    tree._draw_logo_placeholder(
        ax=axis,
        x=2.0,
        y=3.0,
        text='-',
        color='black',
        logo_width=0.25,
        logo_height=0.78,
    )

    assert axis.text_calls == []
    assert len(axis.patch_calls) == 2
    rect = axis.patch_calls[0]
    expected_center_x = 2.0 + (0.25 * tree.AA_LOGO_MISSING_CENTER_SHIFT_RATIO)
    expected_width = (0.25 * tree.AA_LOGO_SLOT_INNER_WIDTH_RATIO) * tree.AA_LOGO_MISSING_BOX_WIDTH_RATIO
    assert pytest.approx(rect.get_x(), rel=0, abs=1e-12) == expected_center_x - (expected_width / 2.0)
    assert pytest.approx(rect.get_y(), rel=0, abs=1e-12) == 3.0 - (0.78 / 2.0)
    assert pytest.approx(rect.get_width(), rel=0, abs=1e-12) == expected_width
    assert pytest.approx(rect.get_height(), rel=0, abs=1e-12) == 0.78
    bar = axis.patch_calls[1]
    expected_bar_width = expected_width * tree.AA_LOGO_MISSING_BAR_WIDTH_RATIO
    expected_bar_height = 0.78 * tree.AA_LOGO_MISSING_BAR_HEIGHT_RATIO
    assert pytest.approx(bar.get_x(), rel=0, abs=1e-12) == expected_center_x - (expected_bar_width / 2.0)
    assert pytest.approx(bar.get_y(), rel=0, abs=1e-12) == 3.0 - (expected_bar_height / 2.0)
    assert pytest.approx(bar.get_width(), rel=0, abs=1e-12) == expected_bar_width
    assert pytest.approx(bar.get_height(), rel=0, abs=1e-12) == expected_bar_height


def test_draw_aa_logo_centers_glyph_within_site_slot():
    axis = _FakeTreeAxis()

    class _FakeBBox:
        x0 = 0.0
        y0 = 0.0
        width = 2.0
        height = 1.0

    class _FakeGlyph:
        def get_extents(self):
            return _FakeBBox()

    class _FakeTextPathModule:
        @staticmethod
        def TextPath(*args, **kwargs):
            return _FakeGlyph()

    class _FakeTransform:
        def __init__(self):
            self.sx = None
            self.sy = None
            self.tx = None
            self.ty = None

        def scale(self, sx, sy):
            self.sx = sx
            self.sy = sy
            return self

        def translate(self, tx, ty):
            self.tx = tx
            self.ty = ty
            return self

        def __add__(self, other):
            return self

    class _FakeTransformsModule:
        @staticmethod
        def Affine2D():
            return _FakeTransform()

    class _FakePathPatch:
        def __init__(self, glyph, transform=None, **kwargs):
            self.glyph = glyph
            self.transform = transform

    class _FakePatchesModule:
        PathPatch = _FakePathPatch

    assert tree._draw_aa_logo(
        ax=axis,
        x=2.0,
        y=3.0,
        probabilities=np.array([1.0], dtype=float),
        orders=np.array(['Q'], dtype=object),
        logo_width=0.5,
        logo_height=1.0,
        mpl_patches=_FakePatchesModule,
        mpl_textpath=_FakeTextPathModule,
        mpl_transforms=_FakeTransformsModule,
        font_properties=None,
    )

    patch = axis.patch_calls[0]
    draw_width = 0.5 * tree.AA_LOGO_SLOT_INNER_WIDTH_RATIO
    left_edge = patch.transform.tx + (_FakeBBox.x0 * patch.transform.sx)
    right_edge = patch.transform.tx + ((_FakeBBox.x0 + _FakeBBox.width) * patch.transform.sx)
    assert pytest.approx(left_edge, rel=0, abs=1e-12) == 2.0 - (draw_width / 2.0)
    assert pytest.approx(right_edge, rel=0, abs=1e-12) == 2.0 + (draw_width / 2.0)


def test_draw_aa_logo_uses_half_width_for_I_glyph():
    axis = _FakeTreeAxis()

    class _FakeBBox:
        x0 = 0.0
        y0 = 0.0
        width = 2.0
        height = 1.0

    class _FakeGlyph:
        def get_extents(self):
            return _FakeBBox()

    class _FakeTextPathModule:
        @staticmethod
        def TextPath(*args, **kwargs):
            return _FakeGlyph()

    class _FakeTransform:
        def __init__(self):
            self.sx = None
            self.sy = None
            self.tx = None
            self.ty = None

        def scale(self, sx, sy):
            self.sx = sx
            self.sy = sy
            return self

        def translate(self, tx, ty):
            self.tx = tx
            self.ty = ty
            return self

        def __add__(self, other):
            return self

    class _FakeTransformsModule:
        @staticmethod
        def Affine2D():
            return _FakeTransform()

    class _FakePathPatch:
        def __init__(self, glyph, transform=None, **kwargs):
            self.glyph = glyph
            self.transform = transform

    class _FakePatchesModule:
        PathPatch = _FakePathPatch

    assert tree._draw_aa_logo(
        ax=axis,
        x=2.0,
        y=3.0,
        probabilities=np.array([1.0], dtype=float),
        orders=np.array(['I'], dtype=object),
        logo_width=0.5,
        logo_height=1.0,
        mpl_patches=_FakePatchesModule,
        mpl_textpath=_FakeTextPathModule,
        mpl_transforms=_FakeTransformsModule,
        font_properties=None,
    )

    patch = axis.patch_calls[0]
    draw_width = 0.5 * tree.AA_LOGO_SLOT_INNER_WIDTH_RATIO * tree.AA_LOGO_CHAR_WIDTH_RATIO['I']
    left_edge = patch.transform.tx + (_FakeBBox.x0 * patch.transform.sx)
    right_edge = patch.transform.tx + ((_FakeBBox.x0 + _FakeBBox.width) * patch.transform.sx)
    assert pytest.approx(left_edge, rel=0, abs=1e-12) == 2.0 - (draw_width / 2.0)
    assert pytest.approx(right_edge, rel=0, abs=1e-12) == 2.0 + (draw_width / 2.0)


def test_expand_highlighted_leaf_ids_to_clade_node_ids_marks_fully_highlighted_clades():
    tr = tree.add_numerical_node_labels(ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1))
    labels = {n.name: int(ete.get_prop(n, "numerical_label")) for n in tr.traverse() if n.name}
    out = tree._expand_highlighted_leaf_ids_to_clade_node_ids(
        tr,
        highlighted_leaf_ids=[labels["A"], labels["B"]],
    )
    assert out == {labels["A"], labels["B"], labels["X"]}


def test_get_logo_glyph_caches_by_character():
    class _FakeTextPathModule:
        call_count = 0

        @classmethod
        def TextPath(cls, *args, **kwargs):
            cls.call_count += 1
            return object()

    glyph1 = tree._get_logo_glyph(_FakeTextPathModule, None, 'Q')
    glyph2 = tree._get_logo_glyph(_FakeTextPathModule, None, 'Q')
    glyph3 = tree._get_logo_glyph(_FakeTextPathModule, None, 'R')

    assert glyph1 is glyph2
    assert glyph1 is not glyph3
    assert _FakeTextPathModule.call_count == 2


def test_format_tree_scale_label_includes_units():
    assert tree._format_tree_scale_label(0.2) == "0.2 subs/codon site"


def test_estimate_text_right_limit_uses_fallback_text_width_ratio():
    axis = _FakeTreeAxis()
    x_right = tree._estimate_text_right_limit(
        ax=axis,
        x_left=-0.1,
        base_right=1.0,
        text_items=[{
            'x': 1.0,
            'text': 'ABCDEFGHIJ',
            'fontsize': tree.TREE_TIP_LABEL_TEXT_SIZE,
            'ha': 'left',
            'fallback_char_ratio': 0.02,
        }],
    )
    expected_ratio = 10 * 0.02
    expected_right = (1.0 - (expected_ratio * -0.1)) / (1.0 - expected_ratio)
    assert pytest.approx(x_right, rel=0, abs=1e-12) == expected_right


def test_ellipsize_middle_preserves_prefix_and_suffix():
    text = "Homo_sapiens_GENEBLAH_ISOFORMBLAH1"
    assert tree._ellipsize_middle(text, 23) == "Homo_sapie...OFORMBLAH1"


def test_fit_leaf_label_items_ellipsizes_when_tree_would_be_too_narrow():
    axis = _FakeTreeAxis()
    long_label = (
        "Homo_sapiens_GENEBLAH_ISOFORMBLAH1_EXTRA_LONG_SUFFIX_0123456789_ABCD_"
        "MORE_TEXT_TO_FORCE_MIDDLE_ELLIPSIS_BEYOND_128_CHARACTERS_AND_KEEP_GOING"
    )
    assert len(long_label) > tree.TREE_TIP_LABEL_NO_ELLIPSIS_UP_TO_CHARS
    leaf_items, x_right = tree._fit_leaf_label_items(
        ax=axis,
        x_left=-0.1,
        content_right=0.8,
        static_text_items=[],
        leaf_label_items=[{
            'x': 0.9,
            'y': 0.0,
            'text': long_label,
            'fontsize': tree.TREE_TIP_LABEL_TEXT_SIZE,
            'color': 'black',
            'va': 'center',
            'ha': 'left',
            'clip_on': False,
            'fallback_char_ratio': 0.024,
        }],
    )
    assert "..." in leaf_items[0]['text']
    assert tree._get_content_width_ratio(-0.1, 0.8, x_right) >= tree.TREE_CONTENT_MIN_WIDTH_RATIO


def test_fit_leaf_label_items_keeps_tps_sized_labels_unshortened():
    axis = _FakeTreeAxis()
    axis.figure = _FakeTreeFigure()
    label = "Adiantum_capillus-veneris_CM043955.1_cds_KAI5070148.1_14541"
    assert len(label) <= tree.TREE_TIP_LABEL_NO_ELLIPSIS_UP_TO_CHARS
    leaf_items, x_right = tree._fit_leaf_label_items(
        ax=axis,
        x_left=-0.1,
        content_right=0.8,
        static_text_items=[],
        leaf_label_items=[{
            'x': 0.9,
            'y': 0.0,
            'text': label,
            'fontsize': tree.TREE_TIP_LABEL_TEXT_SIZE,
            'color': 'black',
            'va': 'center',
            'ha': 'left',
            'clip_on': False,
            'fallback_char_ratio': tree.TREE_FALLBACK_TEXT_CHAR_WIDTH_EM,
        }],
    )
    assert leaf_items[0]['text'] == label
    assert 1.1 < x_right < 1.6


def test_should_use_exact_text_layout_disables_exact_path_for_large_trees():
    assert tree._should_use_exact_text_layout(num_leaves=50, num_text_items=100)
    assert not tree._should_use_exact_text_layout(
        num_leaves=tree.TREE_EXACT_TEXT_LAYOUT_MAX_LEAVES + 1,
        num_text_items=100,
    )
    assert not tree._should_use_exact_text_layout(
        num_leaves=50,
        num_text_items=tree.TREE_EXACT_TEXT_LAYOUT_MAX_ITEMS + 1,
    )
