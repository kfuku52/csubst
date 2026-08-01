import ast
import gzip
import io
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from csubst import sequence_io
from csubst._vendor import pyvolve
from csubst._vendor.pyvolve.evolver import Evolver


def test_runtime_package_has_no_biopython_imports():
    package_root = Path(sequence_io.__file__).resolve().parent
    unexpected = list()
    for path in package_root.rglob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "Bio" or name.startswith("Bio.") for name in names):
                unexpected.append("{}:{}".format(path, node.lineno))
    assert unexpected == []


def test_read_fasta_records_preserves_descriptions_and_uses_first_token_as_id():
    source = io.StringIO(">seq1 description here\r\nAA\r\nAA\r\n>seq2\r\nTTTT\r\n")
    records = sequence_io.read_fasta_records(source)
    assert [(record.description, record.id, record.sequence) for record in records] == [
        ("seq1 description here", "seq1", "AAAA"),
        ("seq2", "seq2", "TTTT"),
    ]


def test_read_fasta_records_removes_sequence_whitespace_like_biopython():
    records = sequence_io.read_fasta_records(
        io.StringIO(">seq1\nAC GT\tAA\n C G \n")
    )
    assert records[0].sequence == "ACGTAACG"


def test_read_fasta_records_rejects_sequence_before_header():
    with pytest.raises(ValueError, match="sequence line appeared before header"):
        sequence_io.read_fasta_records(io.StringIO("AAAA\n>seq1\nTTTT\n"))


def test_records_to_dict_rejects_duplicate_selected_ids():
    records = sequence_io.read_fasta_records(
        io.StringIO(">seq1 first\nAAAA\n>seq1 second\nTTTT\n")
    )
    with pytest.raises(ValueError, match="Duplicate FASTA header"):
        sequence_io.records_to_dict(records, key="id")


def test_write_fasta_records_uses_legacy_60_column_wrapping():
    output = io.StringIO()
    count = sequence_io.write_fasta_records(
        [sequence_io.FastaRecord("seq1", "A" * 65), sequence_io.FastaRecord("empty", "")],
        output,
    )
    assert count == 2
    assert output.getvalue() == ">seq1\n{}\nAAAAA\n>empty\n\n".format("A" * 60)


def test_fasta_io_supports_gzip_paths(tmp_path):
    path = tmp_path / "records.fa.gz"
    sequence_io.write_fasta_dict(OrderedDict([("a", "AAAA"), ("b", "TTTT")]), path)
    with gzip.open(path, mode="rt", encoding="utf-8") as handle:
        assert handle.read() == ">a\nAAAA\n>b\nTTTT\n"
    assert [record.id for record in sequence_io.read_fasta_records(path)] == ["a", "b"]


def test_vendored_pyvolve_read_frequencies_uses_fasta_and_alignment_columns(tmp_path):
    path = tmp_path / "alignment.fa"
    path.write_text(">a\nAC\n>b\nAT\n", encoding="utf-8")
    reader = pyvolve.ReadFrequencies("nucleotide", file=str(path), columns=[2])
    observed = reader.compute_frequencies()
    assert np.allclose(observed, np.array([0.0, 0.5, 0.0, 0.5]))


def test_vendored_pyvolve_read_frequencies_rejects_non_alignment_columns(tmp_path):
    path = tmp_path / "sequences.fa"
    path.write_text(">a\nAC\n>b\nATT\n", encoding="utf-8")
    with pytest.raises(TypeError, match="does not appear to be an.*alignment"):
        pyvolve.ReadFrequencies("nucleotide", file=str(path), columns=[1])


def test_vendored_pyvolve_rejects_non_fasta_input_format(tmp_path):
    path = tmp_path / "alignment.phy"
    path.write_text("2 2\na AC\nb AT\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Only FASTA input"):
        pyvolve.ReadFrequencies("nucleotide", file=str(path), format="phylip")


def test_vendored_pyvolve_evolver_writes_fasta_and_rejects_other_formats(tmp_path):
    evolver = Evolver.__new__(Evolver)
    evolver.seqfile = str(tmp_path / "simulated.fa")
    evolver.seqfmt = "fasta"
    evolver._write_sequences(OrderedDict([("tip1", "A" * 65), ("tip2", "TT")]))
    assert (tmp_path / "simulated.fa").read_text(encoding="utf-8") == (
        ">tip1\n{}\nAAAAA\n>tip2\nTT\n".format("A" * 60)
    )

    evolver.seqfmt = "phylip"
    with pytest.raises(TypeError, match="Only FASTA output"):
        evolver._write_sequences(OrderedDict([("tip1", "AAAA")]))
