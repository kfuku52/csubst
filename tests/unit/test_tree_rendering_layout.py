from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import patches, textpath, transforms
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from csubst import tree


def test_logo_glyphs_are_centered_and_fit_their_slots():
    axis = Figure().subplots()
    widths = []
    for letter in ("Q", "I"):
        assert tree._draw_aa_logo(
            ax=axis, x=2.0, y=3.0, probabilities=np.array([1.0]),
            orders=np.array([letter]), logo_width=0.5, logo_height=1.0,
            mpl_patches=patches, mpl_textpath=textpath, mpl_transforms=transforms,
            font_properties=FontProperties(),
        )
        patch = axis.patches[-1]
        bounds = patch.get_path().transformed(
            patch.get_transform() - axis.transData
        ).get_extents()
        assert (bounds.x0 + bounds.x1) / 2 == pytest.approx(2.0)
        assert (bounds.y0, bounds.y1) == pytest.approx((2.5, 3.5))
        assert 0 < bounds.width <= 0.5
        widths.append(bounds.width)
    assert widths[1] < widths[0]


def test_ellipsize_middle_preserves_prefix_and_suffix():
    text = "Homo_sapiens_GENEBLAH_ISOFORMBLAH1"
    assert tree._ellipsize_middle(text, 23) == "Homo_sapie...OFORMBLAH1"


def test_fit_leaf_label_items_ellipsizes_when_tree_would_be_too_narrow():
    axis = SimpleNamespace(figure=None)
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
    axis = SimpleNamespace(figure=Figure(figsize=(tree.TREE_FIG_WIDTH, 4.0)))
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
