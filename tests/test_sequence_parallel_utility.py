import numpy
import pandas
import pytest

from csubst import foreground
from csubst import genetic_code
from csubst import parallel
from csubst import sequence
from csubst import utility


def test_rgb_to_hex_rounding_matches_manual_conversion():
    # 0.5 * 255 = 127.5 -> 128 after rounding.
    assert utility.rgb_to_hex(0.0, 0.5, 1.0) == "0x0080FF"


def test_rgb_to_hex_rejects_values_larger_than_one():
    with pytest.raises(AssertionError):
        utility.rgb_to_hex(1.1, 0.0, 0.0)


def test_get_chunks_for_python_list():
    chunks, starts = parallel.get_chunks(list(range(10)), threads=3)
    assert [len(c) for c in chunks] == [3, 3, 4]
    assert starts == [0, 3, 6]
    assert chunks[0] == [0, 1, 2]
    assert chunks[2] == [6, 7, 8, 9]


def test_get_chunks_for_numpy_array():
    arr = numpy.arange(10).reshape(5, 2)
    chunks, starts = parallel.get_chunks(arr, threads=2)
    assert [c.shape for c in chunks] == [(2, 2), (3, 2)]
    assert starts == [0, 2]
    numpy.testing.assert_array_equal(chunks[0], arr[:2, :])
    numpy.testing.assert_array_equal(chunks[1], arr[2:, :])


def test_calc_omega_state_matches_manual_tensor_product():
    g = {"state_columns": [[0, 0, 0], [0, 0, 1]]}
    state_nuc = numpy.array(
        [
            [
                [1.0, 0.0],  # codon 1, pos 1
                [1.0, 0.0],  # codon 1, pos 2
                [0.7, 0.3],  # codon 1, pos 3
                [1.0, 0.0],  # codon 2, pos 1
                [1.0, 0.0],  # codon 2, pos 2
                [0.2, 0.8],  # codon 2, pos 3
            ]
        ],
        dtype=float,
    )
    expected = numpy.array([[[0.7, 0.3], [0.2, 0.8]]], dtype=float)
    out = sequence.calc_omega_state(state_nuc=state_nuc, g=g)
    numpy.testing.assert_allclose(out, expected, atol=1e-12)


def test_calc_omega_state_rejects_non_triplet_length():
    g = {"state_columns": [[0, 0, 0]]}
    state_nuc = numpy.zeros((1, 4, 2), dtype=float)
    with pytest.raises(Exception, match="not multiple of 3"):
        sequence.calc_omega_state(state_nuc=state_nuc, g=g)


def test_cdn2pep_state_sums_synonymous_codons():
    g = {
        "amino_acid_orders": numpy.array(["K", "N"]),
        "synonymous_indices": {"K": [0], "N": [1]},
    }
    state_cdn = numpy.array([[[0.7, 0.3], [0.2, 0.8]]], dtype=float)
    out = sequence.cdn2pep_state(state_cdn=state_cdn, g=g)
    numpy.testing.assert_allclose(out, state_cdn, atol=1e-12)


def test_translate_state_handles_missing_states():
    g = {
        "float_tol": 1e-12,
        "codon_orders": numpy.array(["AAA", "AAT"]),
        "amino_acid_orders": numpy.array(["K", "N"]),
        "state_cdn": numpy.array([[[0.1, 0.9], [0.0, 0.0]]], dtype=float),
        "state_pep": numpy.array([[[0.1, 0.9], [0.0, 0.0]]], dtype=float),
    }
    assert sequence.translate_state(0, "codon", g) == "AAT---"
    assert sequence.translate_state(0, "aa", g) == "N-"


def test_get_state_index_with_ambiguity_gap_and_unknown():
    input_state = numpy.array(["AAA", "AAG", "AAT", "AAC"])
    out = sequence.get_state_index("AAN", input_state, genetic_code.ambiguous_table)
    assert sorted(out) == [0, 1, 2, 3]
    assert sequence.get_state_index("-", input_state, genetic_code.ambiguous_table) == []
    assert sequence.get_state_index("XZZ", input_state, genetic_code.ambiguous_table) == []


def test_read_fasta_and_calc_identity(tmp_path):
    fasta = tmp_path / "toy.fa"
    fasta.write_text(">s1\nAA\nAA\n>s2\nAATA\n", encoding="utf-8")
    seqs = sequence.read_fasta(str(fasta))
    assert seqs == {"s1": "AAAA", "s2": "AATA"}
    assert sequence.calc_identity("AATA", "AACA") == 0.75


def test_calc_identity_requires_equal_length():
    with pytest.raises(AssertionError, match="identical"):
        sequence.calc_identity("AAA", "AA")


def test_combinations_count_matches_hand_calculation():
    # 5 choose 2 = 10
    assert foreground.combinations_count(5, 2) == 10
    # Symmetry nCr == nC(n-r)
    assert foreground.combinations_count(8, 6) == foreground.combinations_count(8, 2)
