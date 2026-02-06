import pandas

from csubst import parser_uniprot


def test_normalize_feature_types_defaults_to_all():
    assert parser_uniprot._normalize_feature_types(None) is None
    assert parser_uniprot._normalize_feature_types("all") is None
    assert parser_uniprot._normalize_feature_types("*") is None
    assert parser_uniprot._normalize_feature_types("Active site,Binding site") == ["Active site", "Binding site"]


def test_extract_accession_from_seq_name_patterns():
    assert parser_uniprot._extract_accession_from_seq_name("AF-P04421-F1-model_v2_A") == "P04421"
    assert parser_uniprot._extract_accession_from_seq_name("P04421_A") == "P04421"
    assert parser_uniprot._extract_accession_from_seq_name("NO_ACCESSION") is None


def test_resolve_accession_from_rcsb(monkeypatch):
    def fake_get_json(url, timeout=30):
        if "polymer_entity_instance/2X15/A" in url:
            return {"rcsb_polymer_entity_instance_container_identifiers": {"entity_id": "1"}}
        if "polymer_entity/2X15/1" in url:
            return {"rcsb_polymer_entity_container_identifiers": {"uniprot_ids": ["P00558"]}}
        raise AssertionError(url)

    monkeypatch.setattr(parser_uniprot, "_http_get_json", fake_get_json)
    out = parser_uniprot.resolve_uniprot_accession(seq_name="2X15_A", pdb_id="2X15")
    assert out == "P00558"


def test_get_uniprot_features_parses_locations(monkeypatch):
    def fake_get_json(url, timeout=30):
        return {
            "features": [
                {
                    "type": "Active site",
                    "description": "Catalytic",
                    "location": {"start": {"value": 53}, "end": {"value": 53}},
                },
                {
                    "type": "Binding site",
                    "description": "Ligand",
                    "location": {"start": {"value": 60}, "end": {"value": 62}},
                },
                {
                    "type": "Region",
                    "description": "Bad location",
                    "location": {"start": {"value": None}, "end": {"value": 5}},
                },
            ]
        }

    monkeypatch.setattr(parser_uniprot, "_http_get_json", fake_get_json)
    features = parser_uniprot.get_uniprot_features("P04421")
    assert len(features) == 2
    assert features[0]["type"] == "Active site"
    assert features[1]["start"] == 60
    assert features[1]["end"] == 62


def test_add_uniprot_site_annotations_default_excludes_redundant_info(monkeypatch):
    df = pandas.DataFrame(
        {
            "codon_site_pdb_P04421_A": [1, 2, 3, 4],
            "codon_site_P04421_A": [1, 2, 3, 4],
        }
    )
    g = {"pdb": "AF-P04421-F1-model_v2.pdb", "uniprot_feature_types": None}

    monkeypatch.setattr(parser_uniprot, "resolve_uniprot_accession", lambda seq_name, pdb_id=None: "P04421")
    monkeypatch.setattr(
        parser_uniprot,
        "get_uniprot_features",
        lambda accession: [
            {"type": "Chain", "description": "Example protein", "start": 1, "end": 4},
            {"type": "Topological domain", "description": "Cytoplasmic", "start": 1, "end": 2},
            {"type": "Helix", "description": "", "start": 3, "end": 3},
        ],
    )

    out = parser_uniprot.add_uniprot_site_annotations(df=df, g=g)
    assert "uniprot_acc_P04421_A" not in out.columns
    assert "uniprot_feature_types_P04421_A" not in out.columns
    assert out.loc[0, "uniprot_feature_descriptions_P04421_A"].startswith("Topological domain")
    assert out.loc[2, "uniprot_feature_descriptions_P04421_A"].startswith("Helix")
    assert "Chain" not in out.loc[0, "uniprot_feature_descriptions_P04421_A"]
    assert out.loc[3, "uniprot_feature_count_P04421_A"] == 0


def test_add_uniprot_site_annotations_can_keep_redundant_info(monkeypatch):
    df = pandas.DataFrame(
        {
            "codon_site_pdb_P04421_A": [1, 2],
            "codon_site_P04421_A": [1, 2],
        }
    )
    g = {
        "pdb": "AF-P04421-F1-model_v2.pdb",
        "uniprot_feature_types": None,
        "uniprot_include_redundant": True,
    }

    monkeypatch.setattr(parser_uniprot, "resolve_uniprot_accession", lambda seq_name, pdb_id=None: "P04421")
    monkeypatch.setattr(
        parser_uniprot,
        "get_uniprot_features",
        lambda accession: [
            {"type": "Chain", "description": "Example protein", "start": 1, "end": 2},
        ],
    )

    out = parser_uniprot.add_uniprot_site_annotations(df=df, g=g)
    assert "uniprot_acc_P04421_A" in out.columns
    assert out.loc[0, "uniprot_feature_types_P04421_A"] == "Chain"
    assert "Chain: Example protein (1-2)" in out.loc[0, "uniprot_feature_descriptions_P04421_A"]


def test_add_uniprot_site_annotations_with_feature_filter(monkeypatch):
    df = pandas.DataFrame(
        {
            "codon_site_pdb_P04421_A": [53, 60],
            "codon_site_P04421_A": [53, 60],
        }
    )
    g = {"pdb": "AF-P04421-F1-model_v2.pdb", "uniprot_feature_types": ["Active site"]}

    monkeypatch.setattr(parser_uniprot, "resolve_uniprot_accession", lambda seq_name, pdb_id=None: "P04421")
    monkeypatch.setattr(
        parser_uniprot,
        "get_uniprot_features",
        lambda accession: [
            {"type": "Active site", "description": "Catalytic", "start": 53, "end": 53},
            {"type": "Binding site", "description": "Ligand", "start": 60, "end": 61},
        ],
    )

    out = parser_uniprot.add_uniprot_site_annotations(df=df, g=g)
    assert out.loc[0, "uniprot_feature_count_P04421_A"] == 1
    assert "uniprot_feature_types_P04421_A" not in out.columns
    assert out.loc[0, "uniprot_feature_descriptions_P04421_A"].startswith("Active site")
    assert out.loc[1, "uniprot_feature_count_P04421_A"] == 0
    assert out.loc[1, "uniprot_feature_descriptions_P04421_A"] == ""
