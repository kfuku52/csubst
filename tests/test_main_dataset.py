import gzip

from csubst import main_dataset


def test_copy_dataset_files_writes_fasta_as_gz(tmp_path):
    dataset_dir = tmp_path / "dataset"
    out_dir = tmp_path / "out"
    dataset_dir.mkdir()
    out_dir.mkdir()
    (dataset_dir / "PEPC.alignment.fa").write_text(">s1\nAAAA\n>s2\nAAAT\n", encoding="utf-8")
    (dataset_dir / "PEPC.untrimmed_cds.fa").write_text(">s1\nATGATG\n>s2\nATGATA\n", encoding="utf-8")
    (dataset_dir / "PEPC.tree.nwk").write_text("(s1:1,s2:1);\n", encoding="utf-8")
    (dataset_dir / "PEPC.foreground.txt").write_text("lineage\ts1\n", encoding="utf-8")
    (dataset_dir / "PEPC.alignment.fa.state").write_text("# state\n", encoding="utf-8")
    (dataset_dir / "PEPC.alignment.fa.treefile").write_text("(s1:1,s2:1);\n", encoding="utf-8")
    # Different dataset prefix should be ignored.
    (dataset_dir / "PEPC2.alignment.fa").write_text(">x\nAAAA\n", encoding="utf-8")

    main_dataset._copy_dataset_files(name="PEPC", dir_dataset=str(dataset_dir), output_dir=str(out_dir))

    assert (out_dir / "alignment.fa.gz").exists() is True
    assert (out_dir / "untrimmed_cds.fa.gz").exists() is True
    assert (out_dir / "tree.nwk").exists() is True
    assert (out_dir / "foreground.txt").exists() is True
    assert (out_dir / "csubst_iqtree" / "alignment.fa.state").exists() is True
    assert (out_dir / "csubst_iqtree" / "alignment.fa.treefile").exists() is True
    assert (out_dir / "alignment.fa.state").exists() is False
    assert (out_dir / "PEPC2.alignment.fa.gz").exists() is False

    with gzip.open(out_dir / "alignment.fa.gz", mode="rt", encoding="utf-8") as f:
        assert f.read() == ">s1\nAAAA\n>s2\nAAAT\n"
    with gzip.open(out_dir / "untrimmed_cds.fa.gz", mode="rt", encoding="utf-8") as f:
        assert f.read() == ">s1\nATGATG\n>s2\nATGATA\n"
    assert (out_dir / "tree.nwk").read_text(encoding="utf-8") == "(s1:1,s2:1);\n"
