from pathlib import Path
import types

import pytest

from csubst import local_sequence_search


def test_local_blastp_uses_xml2_and_disables_usage_reporting(monkeypatch):
    def run(command, *, input_text, env):
        assert command == [
            'blastp', '-db', '"/db with spaces/swissprot"', '-query', '-', '-outfmt', '16',
            '-evalue', '0.01', '-max_target_seqs', '50', '-num_threads', '2',
        ]
        assert input_text == '>query\nACDEFGHIK\n'
        assert env['BLAST_USAGE_REPORT'] == 'false'
        assert env['NCBI_CONFIG__BLAST__DATA_LOADERS'] == 'none'
        return types.SimpleNamespace(stdout=(
            '<BlastXML2><Hit><HitDescr><accession>P69905</accession></HitDescr></Hit></BlastXML2>'
        ))

    monkeypatch.setenv('BLAST_USAGE_REPORT', 'true')
    monkeypatch.setattr(local_sequence_search, '_run', run)
    assert local_sequence_search.blastp_accessions('ACDEFGHIK', '/db with spaces/swissprot', 0.01, 2) == ['P69905']


def test_local_mmseqs_ranks_unique_pdb_entries_and_removes_workdir(tmp_path, monkeypatch):
    database = str(tmp_path / 'pdb')
    (tmp_path / 'pdb.dbtype').touch()
    workdirs = []

    def run(command):
        assert command[:2] == ['mmseqs', 'easy-search']
        query, output = Path(command[2]), Path(command[4])
        workdirs.append(query.parent)
        assert query.read_text() == '>query\nACDEFGHIK\n'
        assert command[3] == database
        assert command[6:] == [
            '--search-type', '1', '--alignment-mode', '3', '-e', '0.1',
            '--min-seq-id', '0.25', '--threads', '2', '--format-output', 'target,evalue,bits,theader',
        ]
        output.write_text(
            '4HHB_A\t1e-10\t30\t4HHB_A a chain\n'
            '4HHB_2\t1e-30\t40\t4HHB_2 an entity\n'
            '000000001\t1e-20\t50\tpdb_10001XYZ.A extended entry\n'
            '2or1\t1e-20\t60\tpdb|2OR1|A pipe ID\n'
        )

    monkeypatch.setattr(local_sequence_search, '_run', run)
    assert local_sequence_search.mmseqs_pdb_ids('ACDEFGHIK', database, 0.1, 0.25, 2) == [
        '4hhb', '2or1', 'pdb_10001xyz',
    ]
    assert len(workdirs) == 1 and not workdirs[0].exists()


def test_local_mmseqs_rejects_missing_dbtype(tmp_path):
    with pytest.raises(ValueError, match=r'missing \.dbtype'):
        local_sequence_search.mmseqs_pdb_ids('ACDEFGHIK', str(tmp_path / 'missing'), 1, 0.25, 1)
