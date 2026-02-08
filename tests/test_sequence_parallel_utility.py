import numpy
import pandas
import pytest

from csubst import foreground
from csubst import genetic_code
from csubst import parallel
from csubst import sequence
from csubst import utility


def _starmap_add_mul(a, b):
    return (a + b) * 2


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


def test_get_chunks_avoids_empty_chunks_when_threads_exceed_items():
    chunks, starts = parallel.get_chunks([10, 11], threads=8)
    assert chunks == [[10], [11]]
    assert starts == [0, 1]


def test_get_chunks_supports_finer_chunk_factor_splitting():
    chunks, starts = parallel.get_chunks(list(range(10)), threads=2, chunk_factor=3)
    assert [len(c) for c in chunks] == [1, 1, 2, 2, 2, 2]
    assert starts == [0, 1, 2, 4, 6, 8]


def test_resolve_n_jobs_is_capped_by_workload_size():
    assert parallel.resolve_n_jobs(num_items=0, threads=8) == 1
    assert parallel.resolve_n_jobs(num_items=2, threads=8) == 2
    assert parallel.resolve_n_jobs(num_items=8, threads=2) == 2


def test_resolve_parallel_backend_auto_uses_task_policy():
    g = {"parallel_backend": "auto"}
    assert parallel.resolve_parallel_backend(g=g, task="general") == "multiprocessing"
    assert parallel.resolve_parallel_backend(g=g, task="reducer") == "multiprocessing"


def test_run_starmap_single_process_matches_direct_evaluation():
    args = [(1, 2), (3, 4), (5, 6)]
    out = parallel.run_starmap(_starmap_add_mul, args, n_jobs=1, backend="multiprocessing")
    assert out == [6, 14, 22]


def test_run_starmap_process_backend_preserves_order():
    args = [(1, 2), (3, 4), (5, 6), (7, 8)]
    out = parallel.run_starmap(_starmap_add_mul, args, n_jobs=2, backend="multiprocessing")
    assert out == [6, 14, 22, 30]


def test_run_starmap_threading_backend_works():
    args = [(2, 3), (4, 5), (6, 7)]
    out = parallel.run_starmap(_starmap_add_mul, args, n_jobs=2, backend="threading")
    assert out == [10, 18, 26]


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


def test_cdn2pep_state_selected_branch_ids_keeps_global_branch_index():
    g = {
        "amino_acid_orders": numpy.array(["K", "N"]),
        "synonymous_indices": {"K": [0, 1], "N": [2]},
    }
    state_cdn = numpy.array(
        [
            [[0.2, 0.3, 0.5], [0.0, 0.0, 1.0]],
            [[0.1, 0.4, 0.5], [0.6, 0.2, 0.2]],
            [[0.3, 0.3, 0.4], [0.4, 0.1, 0.5]],
            [[0.9, 0.0, 0.1], [0.2, 0.3, 0.5]],
        ],
        dtype=float,
    )
    selected = numpy.array([1, 3], dtype=numpy.int64)
    out = sequence.cdn2pep_state(state_cdn=state_cdn, g=g, selected_branch_ids=selected)
    expected_selected = numpy.zeros((2, 2, 2), dtype=float)
    expected_selected[:, :, 0] = state_cdn[selected, :, :][:, :, [0, 1]].sum(axis=2)
    expected_selected[:, :, 1] = state_cdn[selected, :, 2]
    numpy.testing.assert_allclose(out[selected, :, :], expected_selected, atol=1e-12)
    assert out[0, :, :].sum() == 0
    assert out[2, :, :].sum() == 0


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
