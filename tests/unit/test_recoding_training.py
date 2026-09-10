import numpy as np
import pytest

from csubst import genetic_code, recoding


def setup_training(tmp_path):
    table = genetic_code.get_codon_table(1)
    codons = [codon for aa, codon in table if aa != '*']
    source = tmp_path / 'train.fa'
    source.write_text('>a\n' + ''.join(codons) + '\n>b\n' + ''.join(codons[::-1]) + '\n')
    target = tmp_path / 'test.fa'
    target.write_text('>a\n' + 'GCT' * 10 + '\n')
    return dict(alignment_file=str(target), nonsyn_recode_training_alignment=str(source),
                codon_table=table, amino_acid_orders=np.array(list('ACDEFGHIKLMNPQRSTVWY')),
                nonsyn_recode_seed=7, nonsyn_recode_random_starts=4, threads=1), source, target


@pytest.mark.parametrize('scheme', ['srchisq6', 'kgbauto6'])
def test_independent_training_grouping_is_invariant_to_target_alignment(tmp_path, scheme):
    g, _, target = setup_training(tmp_path)
    before = recoding._build_auto_recoded_groups(g, scheme)
    target.write_text('>a\n' + 'TTT' * 20 + '\n')
    after = recoding._build_auto_recoded_groups(g, scheme)
    assert before == after
    assert g['_nonsyn_recode_training_provenance']['source'] == 'separate_alignment'


def test_overwriting_training_filename_invalidates_cache(tmp_path):
    g, source, _ = setup_training(tmp_path)
    orders = g['amino_acid_orders']
    before = recoding._get_alignment_aa_statistics(g, orders)[0].copy()
    fingerprint = g['_nonsyn_recode_training_provenance']['sha256']
    source.write_text('>a\n' + 'GCT' * 10 + '\n')
    after = recoding._get_alignment_aa_statistics(g, orders)[0]
    assert before.shape != after.shape
    assert fingerprint != g['_nonsyn_recode_training_provenance']['sha256']


def test_genetic_code_change_invalidates_cache(tmp_path):
    g, _, _ = setup_training(tmp_path)
    orders = g['amino_acid_orders']
    before = recoding._get_alignment_aa_statistics(g, orders)[0].copy()
    g['codon_table'] = [(('A' if codon == 'TTT' else aa), codon) for aa, codon in g['codon_table']]
    after = recoding._get_alignment_aa_statistics(g, orders)[0]
    assert np.count_nonzero(before != after) == 2
