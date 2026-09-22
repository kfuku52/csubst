import gzip
import io
from collections import OrderedDict

import numpy as np
import pytest

from csubst import sequence_io
from csubst import parser_iqtree
from csubst._vendor import pyvolve
from csubst._vendor.pyvolve.evolver import Evolver


def test_read_fasta_records_preserves_descriptions_and_uses_first_token_as_id():
    source = io.StringIO(">seq1 description here\r\nAA\r\nAA\r\n>seq2\r\nTTTT\r\n")
    records = sequence_io.read_fasta_records(source)
    assert [(record.description, record.id, record.sequence) for record in records] == [
        ("seq1 description here", "seq1", "AAAA"),
        ("seq2", "seq2", "TTTT"),
    ]


@pytest.mark.parametrize('compressed', [False, True])
def test_alignment_site_count_uses_fasta_whitespace_rules(tmp_path, compressed):
    path = tmp_path / ('alignment.fa.gz' if compressed else 'alignment.fa')
    opener = gzip.open if compressed else open
    with opener(path, 'wt', encoding='utf-8', newline='') as handle:
        handle.write('>A\n AT G\t\r\nGC T\r\n>B\nATGCCC\n')
    assert sequence_io.read_fasta_records(path)[0].sequence == 'ATGGCT'
    assert parser_iqtree._infer_num_input_site_from_alignment_file(path) == 2


def test_fasta_iterator_stops_before_reading_second_sequence():
    class FirstRecordOnly(io.StringIO):
        def __next__(self):
            line = super().__next__()
            if line.startswith('SHOULD_NOT_READ'):
                raise AssertionError('Read beyond the requested record')
            return line

    source = FirstRecordOnly('>A\nATG GCT\n>B\nSHOULD_NOT_READ\n')
    records = sequence_io.iter_fasta_records(source)
    assert next(records).sequence == 'ATGGCT'
    records.close()
    assert not source.closed


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
