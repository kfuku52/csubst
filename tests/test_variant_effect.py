import numpy as np
import pandas as pd
import pytest

from csubst import ete
from csubst import main_sites
from csubst import tree
from csubst import variant_effect


def _two_tip_context(tmp_path, threshold=0.8):
    tree_obj = tree.add_numerical_node_labels(
        ete.PhyloNode("(A:1,B:1)R;", format=1)
    )
    node_id = {
        str(node.name): int(ete.get_prop(node, "numerical_label"))
        for node in tree_obj.traverse()
    }
    state_pep = np.zeros((len(node_id), 2, 2), dtype=float)
    state_pep[:, :, 0] = 1.0
    state_pep[node_id["R"], 0, :] = [0.9, 0.1]
    state_pep[node_id["A"], 0, :] = [0.1, 0.9]
    alignment = tmp_path / "full_cds.fa"
    alignment.write_text(">A\nGTTGCT\n>B\nGCTGCT\n", encoding="utf-8")
    g = {
        "tree": tree_obj,
        "state_pep": state_pep,
        "amino_acid_orders": np.array(["A", "V"], dtype=object),
        "alignment_file": str(alignment),
        "site_index_alignment": np.array([0, 1], dtype=np.int64),
        "vep_min_event_pp": threshold,
        "float_tol": 1e-12,
    }
    return g, node_id


def test_prepare_context_and_extract_atomic_event_uses_inclusive_pp_threshold(tmp_path):
    g, node_id = _two_tip_context(tmp_path=tmp_path, threshold=0.81)
    bundle = variant_effect.prepare_ancestral_contexts(g=g)
    events = variant_effect.extract_atomic_aa_events(g=g, branch_ids=[node_id["A"]])

    assert bundle["contexts"][node_id["R"]]["sequence"] == "AA"
    assert events.shape[0] == 1
    event = events.iloc[0]
    assert event["event_id"] == "b{}.a1.A>V".format(node_id["A"])
    assert event["state_change"] == "A1V"
    assert event["codon_site_alignment"] == 1
    assert event["aa_position_ancestral"] == 1
    assert event["event_pp"] == pytest.approx(0.81)
    assert event["_context_sequence"] == "AA"


def test_extract_atomic_event_excludes_values_below_threshold(tmp_path):
    g, node_id = _two_tip_context(tmp_path=tmp_path, threshold=0.8100001)
    variant_effect.prepare_ancestral_contexts(g=g)
    events = variant_effect.extract_atomic_aa_events(g=g, branch_ids=[node_id["A"]])
    assert events.empty
    assert events.columns.tolist() == list(variant_effect.EVENT_COLUMNS) + ["_context_sequence"]


def test_extract_atomic_event_respects_filtered_to_alignment_site_mapping(tmp_path):
    g, node_id = _two_tip_context(tmp_path=tmp_path, threshold=0.8)
    variant_effect.prepare_ancestral_contexts(g=g)
    g["state_pep"] = g["state_pep"][:, :1, :]
    g["site_index_alignment"] = np.array([0], dtype=np.int64)
    events = variant_effect.extract_atomic_aa_events(g=g, branch_ids=[node_id["A"]])
    assert events["codon_site_alignment"].tolist() == [1]
    assert events["aa_position_ancestral"].tolist() == [1]


def test_infer_ancestral_gap_presence_uses_parent_state_for_internal_ties():
    tree_obj = tree.add_numerical_node_labels(
        ete.PhyloNode("((A:1,B:1)X:1,C:1)R;", format=1)
    )
    node_id = {
        str(node.name): int(ete.get_prop(node, "numerical_label"))
        for node in tree_obj.traverse()
    }
    presence = variant_effect.infer_ancestral_gap_presence(
        tree_obj=tree_obj,
        presence_by_tip={
            "A": np.array([True]),
            "B": np.array([False]),
            "C": np.array([False]),
        },
        num_node=len(node_id),
        num_site=1,
    )
    assert presence[node_id["R"], 0] == np.bool_(False)
    assert presence[node_id["X"], 0] == np.bool_(False)
    assert presence[node_id["A"], 0] == np.bool_(True)


def _scored_events():
    rows = [
        {
            "event_id": "b1.a1.A>V",
            "branch_id": 1,
            "site": 0,
            "state_change": "A1V",
            "event_pp": 0.9,
            "vesm_llr": -2.0,
        },
        {
            "event_id": "b2.a1.A>G",
            "branch_id": 2,
            "site": 0,
            "state_change": "A1G",
            "event_pp": 0.8,
            "vesm_llr": 1.0,
        },
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("aggregate", "expected"),
    [
        ("most_deleterious", -2.0),
        ("mean", -0.5),
        ("pp_weighted_mean", (-2.0 * 0.9 + 1.0 * 0.8) / 1.7),
    ],
)
def test_attach_scores_to_site_table_supports_structure_aggregates(aggregate, expected):
    g = {"vep_site_aggregate": aggregate}
    out = variant_effect.attach_scores_to_site_table(
        df=pd.DataFrame({"codon_site_alignment": [1, 2]}),
        events=_scored_events(),
        branch_ids=np.array([1, 2], dtype=np.int64),
        g=g,
    )
    assert out.at[0, "vesm_1_state_change"] == "A1V"
    assert out.at[0, "vesm_1_event_pp"] == pytest.approx(0.9)
    assert out.at[0, "vesm_2_llr"] == pytest.approx(1.0)
    assert out.at[0, "vesm_structure_llr"] == pytest.approx(expected)
    assert out.at[0, "vesm_structure_event_count"] == 2
    assert g["_vep_color_limit"] == pytest.approx(abs(expected))


def test_add_structure_coordinates_to_events_keeps_long_event_rows():
    events = _scored_events()
    events.loc[:, "codon_site_alignment"] = 1
    events.loc[:, "_context_sequence"] = "AAAA"
    site_df = pd.DataFrame(
        {
            "codon_site_alignment": [1, 2],
            "codon_site_pdb_obj_A": [42, 43],
        }
    )
    out = variant_effect.add_structure_coordinates_to_events(events=events, site_df=site_df)
    assert out["codon_site_pdb_obj_A"].tolist() == [42, 42]
    assert "_context_sequence" not in out.columns


def test_vesm_plot_width_uses_one_point_gap_between_rotated_tick_labels():
    fig_width,_fig_height,tree_width,site_width = main_sites._get_vesm_plot_dimensions(
        num_sites=48,
        leaf_count=5,
        num_branches=3,
        max_height=10,
    )
    target_pitch_points = 48 * (main_sites.font_size + main_sites.VESM_XTICK_LABEL_GAP_POINTS)
    assert site_width * 0.92 * 72 == pytest.approx(target_pitch_points)
    assert fig_width == pytest.approx(tree_width + site_width + 0.5)
    assert fig_width < 11


def test_plot_vesm_tree_site_writes_plot_and_plot_table(tmp_path, monkeypatch):
    tree_obj = tree.add_numerical_node_labels(
        ete.PhyloNode("(A:1,B:1)R;", format=1)
    )
    node_id = {
        str(node.name): int(ete.get_prop(node, "numerical_label"))
        for node in tree_obj.traverse()
    }
    events = _scored_events().iloc[[0], :].copy()
    events.loc[:, "branch_id"] = node_id["A"]
    events.loc[:, "codon_site_alignment"] = 1
    events.loc[:, "from_aa"] = "A"
    events.loc[:, "to_aa"] = "V"
    g = {
        "tree": tree_obj,
        "branch_ids": np.array([node_id["A"]], dtype=np.int64),
        "mode": "lineage",
        "tree_site_plot_format": "png",
        "tree_site_plot_max_sites": 30,
        "tree_site_fig_max_height": 10,
        "vep_plot": True,
        "vep_min_event_pp": 0.8,
        "_vep_color_limit": 2.0,
        "float_format": "%.6g",
    }
    outbase = str(tmp_path / "example")
    closed_figures = []
    real_close = main_sites.plt.close
    monkeypatch.setattr(main_sites.plt, "close", lambda fig: closed_figures.append(fig))
    output_paths = main_sites.plot_vesm_tree_site(
        events=events,
        df=pd.DataFrame(
            {
                "codon_site_alignment": [1],
                "codon_site_pdb_obj_A": [42],
            }
        ),
        g=g,
        outbase=outbase,
    )
    assert output_paths == [
        outbase + ".vesm_tree_site.png",
        outbase + ".vesm_tree_site.tsv",
    ]
    assert all((tmp_path / path.split("/")[-1]).stat().st_size > 0 for path in output_paths)
    plot_table = pd.read_csv(output_paths[1], sep="\t")
    assert bool(plot_table.at[0, "is_plotted"])
    assert plot_table.at[0, "plot_order"] == 1
    branch_colors = main_sites._get_vesm_branch_color_by_id(
        g=g,
        branch_ids=g["branch_ids"],
    )
    figure = next(item for item in closed_figures if hasattr(item, "axes"))
    tree_axis = figure.axes[0]
    grid_axis = figure.axes[1]
    assert branch_colors[node_id["A"]] in [line.get_color() for line in tree_axis.lines]
    assert grid_axis.get_yticklabels()[0].get_color() == branch_colors[node_id["A"]]
    real_close(figure)
