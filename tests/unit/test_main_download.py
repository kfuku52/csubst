import pytest

from csubst import main_download


def test_normalize_resources_supports_all_and_rejects_unknown():
    assert main_download._normalize_resources('all') == ['vesm-35m', 'prostt5']
    assert main_download._normalize_resources('VESM_35M') == ['vesm-35m']
    with pytest.raises(ValueError, match='resource should be one of'):
        main_download._normalize_resources('unknown')


def test_main_download_prepares_both_resources(monkeypatch, capsys):
    calls = {}

    def ensure_vesm35m_resource(**kwargs):
        calls['vesm'] = kwargs
        return {'resource_dir': '/cache/vesm'}

    def ensure_prostt5_model_files(g):
        calls['prostt5'] = g
        return '/cache/prostt5'

    monkeypatch.setattr(
        main_download.model_resources,
        'ensure_vesm35m_resource',
        ensure_vesm35m_resource,
    )
    monkeypatch.setattr(
        main_download.structural_alphabet,
        'ensure_prostt5_model_files',
        ensure_prostt5_model_files,
    )
    main_download.main_download(
        {
            'resource': 'all',
            'resource_cache_dir': '/cache',
            'resource_lock_poll': 2.0,
            'resource_lock_timeout': 20.0,
            'no_download': True,
        }
    )

    assert calls['vesm']['cache_dir'] == '/cache'
    assert calls['vesm']['no_download'] is True
    assert calls['prostt5']['prostt5_no_download'] is True
    assert 'VESM-35M model files are ready' in capsys.readouterr().out


@pytest.mark.parametrize(
    'key,value,expected',
    [
        ('resource_lock_poll', 0, 'resource_lock_poll'),
        ('resource_lock_timeout', -1, 'resource_lock_timeout'),
    ],
)
def test_main_download_validates_lock_timing(key, value, expected):
    config = {'resource': 'vesm-35m', key: value}
    with pytest.raises(ValueError, match=expected):
        main_download.main_download(config)


@pytest.mark.parametrize('error_type', [FileNotFoundError, ImportError])
def test_main_download_translates_expected_vesm_errors(monkeypatch, error_type):
    def fail(**_kwargs):
        raise error_type('resource unavailable')

    monkeypatch.setattr(main_download.model_resources, 'ensure_vesm35m_resource', fail)
    with pytest.raises(ValueError, match='resource unavailable'):
        main_download.main_download({'resource': 'vesm-35m'})
