import gzip

import pytest

from csubst import main_dataset
from csubst import runtime


@pytest.mark.parametrize(
    "file_name, destination, compress",
    [
        ("alignment.fa", "alignment.fa.gz", True),
        ("sequences.FASTA", "sequences.FASTA.gz", True),
        ("sequences.faa", "sequences.faa.gz", True),
        ("sequences.fna", "sequences.fna.gz", True),
        ("sequences.FA.GZ", "sequences.FA.GZ", False),
        ("alignment.fa.iqtree", "{iqtree_prefix}.iqtree", False),
        ("alignment.fa.log", "{iqtree_prefix}.log", False),
        ("alignment.fa.rate", "{iqtree_prefix}.rate", False),
        ("ALIGNMENT.FA.STATE", "{iqtree_prefix}.state", False),
        ("alignment.fa.treefile", "{iqtree_prefix}.treefile", False),
        ("other.state", "other.state", False),
        ("foreground.txt", "foreground.txt", False),
    ],
)
def test_copy_dataset_file_routing(tmp_path, capsys, file_name, destination, compress):
    dataset_dir = tmp_path / "dataset"
    out_dir = tmp_path / "out"
    dataset_dir.mkdir()
    out_dir.mkdir()
    content = b">s1\nATG\n"
    if file_name.lower().endswith(".gz"):
        content = gzip.compress(content, mtime=0)
    (dataset_dir / ("PGK." + file_name)).write_bytes(content)

    main_dataset._copy_dataset_files("PGK", dataset_dir, out_dir)

    iqtree_prefix = runtime.infer_iqtree_output_prefix(
        alignment_file=str(out_dir / "alignment.fa.gz"),
        iqtree_outdir=str(out_dir / "csubst_iqtree"),
        base_dir=str(out_dir),
    )
    destination = destination.format(
        iqtree_prefix=main_dataset.os.path.relpath(iqtree_prefix, out_dir)
    )
    output = out_dir / destination
    assert sorted(str(path.relative_to(out_dir)) for path in out_dir.rglob("*") if path.is_file()) == [destination]
    assert (gzip.decompress(output.read_bytes()) if compress else output.read_bytes()) == content
    assert capsys.readouterr().out == "Copying PGK file: {}\n".format(destination)


@pytest.mark.parametrize("existing_kind", ["file", "directory", "symlink"])
@pytest.mark.parametrize("force", [False, True])
def test_copy_dataset_preflights_all_destinations(tmp_path, capsys, existing_kind, force):
    dataset_dir = tmp_path / "dataset"
    out_dir = tmp_path / "out"
    dataset_dir.mkdir()
    out_dir.mkdir()
    (dataset_dir / "PGK.a.txt").write_bytes(b"first")
    (dataset_dir / "PGK.z.txt").write_bytes(b"last")
    existing = out_dir / "z.txt"
    if existing_kind == "file":
        existing.write_bytes(b"previous")
    elif existing_kind == "directory":
        existing.mkdir()
    else:
        existing.symlink_to(tmp_path / "missing")

    if existing_kind == "file" and force:
        main_dataset._copy_dataset_files("PGK", dataset_dir, out_dir, force=force)
        assert (out_dir / "a.txt").read_bytes() == b"first"
        assert existing.read_bytes() == b"last"
        assert capsys.readouterr().out == "Copying PGK file: a.txt\nCopying PGK file: z.txt\n"
    else:
        message = "--force yes" if existing_kind == "file" else "not a regular file"
        with pytest.raises(FileExistsError, match=message):
            main_dataset._copy_dataset_files("PGK", dataset_dir, out_dir, force=force)
        assert not (out_dir / "a.txt").exists()
        assert capsys.readouterr().out == ""
        if existing_kind == "file":
            assert existing.read_bytes() == b"previous"
        elif existing_kind == "directory":
            assert existing.is_dir()
        else:
            assert existing.is_symlink()


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
    (dataset_dir / "PEPC.alignment.fa.iqtree").write_text(
        "Model of substitution: MG\n", encoding="utf-8"
    )
    # Different dataset prefix should be ignored.
    (dataset_dir / "PEPC2.alignment.fa").write_text(">x\nAAAA\n", encoding="utf-8")

    main_dataset._copy_dataset_files(name="PEPC", dir_dataset=str(dataset_dir), output_dir=str(out_dir))

    assert (out_dir / "alignment.fa.gz").exists() is True
    assert (out_dir / "untrimmed_cds.fa.gz").exists() is True
    assert (out_dir / "tree.nwk").exists() is True
    assert (out_dir / "foreground.txt").exists() is True
    iqtree_prefix = runtime.infer_iqtree_output_prefix(
        alignment_file=str(out_dir / "alignment.fa.gz"),
        iqtree_outdir=str(out_dir / "csubst_iqtree"),
        base_dir=str(out_dir),
    )
    assert (out_dir / "csubst_iqtree").exists() is True
    assert main_dataset.os.path.isfile(iqtree_prefix + ".state") is True
    assert main_dataset.os.path.isfile(iqtree_prefix + ".treefile") is True
    assert main_dataset.os.path.isfile(iqtree_prefix + ".state.csubst-manifest.json") is True
    assert (out_dir / "alignment.fa.state").exists() is False
    assert (out_dir / "PEPC2.alignment.fa.gz").exists() is False

    with gzip.open(out_dir / "alignment.fa.gz", mode="rt", encoding="utf-8") as f:
        assert f.read() == ">s1\nAAAA\n>s2\nAAAT\n"
    with gzip.open(out_dir / "untrimmed_cds.fa.gz", mode="rt", encoding="utf-8") as f:
        assert f.read() == ">s1\nATGATG\n>s2\nATGATA\n"
    assert (out_dir / "tree.nwk").read_text(encoding="utf-8") == "(s1:1,s2:1);\n"

    with pytest.raises(FileExistsError, match="--force"):
        main_dataset._copy_dataset_files(
            name="PEPC",
            dir_dataset=str(dataset_dir),
            output_dir=str(out_dir),
        )
    main_dataset._copy_dataset_files(
        name="PEPC",
        dir_dataset=str(dataset_dir),
        output_dir=str(out_dir),
        force=True,
    )
