import pytest
from cli_runner import run_csubst as _run_csubst


@pytest.mark.parametrize(
    ("args", "subcommand_log_dir"),
    [
        (("--help",), None),
        (("inspect", "-h"), "csubst_inspect"),
        (("sites", "--help-advanced"), "csubst_sites"),
        (("analyze", "-h"), "csubst_search"),
    ],
)
def test_help_has_no_log_side_effect(tmp_path, args, subcommand_log_dir):
    result = _run_csubst(list(args), cwd=tmp_path)
    assert result.returncode == 0
    assert "CSUBST start:" not in result.stdout
    assert "usage:" in result.stdout.lower()
    assert not (tmp_path / "csubst.log").exists()
    if subcommand_log_dir is not None:
        assert not (tmp_path / subcommand_log_dir / "csubst.log").exists()
    if args == ("sites", "--help-advanced"):
        assert "--expected_state_backend" in result.stdout
        assert "--parallel_" not in result.stdout


def test_cli_version_has_no_log_side_effect(tmp_path):
    result = _run_csubst(["--version"], cwd=tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip().startswith("CSUBST version: ")
    assert "CSUBST start:" not in result.stdout
    assert not (tmp_path / "csubst.log").exists()


@pytest.mark.parametrize("command", ["search", "analyze"])
def test_cli_writes_stderr_output_to_csubst_log(tmp_path, command):
    result = _run_csubst([command, "--does_not_exist"], cwd=tmp_path)
    assert result.returncode != 0
    log_file = tmp_path / "csubst_search" / "csubst.log"
    assert log_file.exists()
    log_text = log_file.read_text(encoding="utf-8")
    assert "error:" in result.stderr.lower()
    assert "--does_not_exist" in result.stderr
    assert "error:" in log_text.lower()
    assert "--does_not_exist" in log_text


@pytest.mark.parametrize(
    "args",
    [
        ("--log_file", "custom-help.log", "--help"),
        ("inspect", "--log_file", "inspect.log", "-h"),
    ],
)
def test_help_ignores_custom_log_file(tmp_path, args):
    custom_log = tmp_path / "logs" / args[-2]
    argv = [str(custom_log) if token == args[-2] else token for token in args]
    result = _run_csubst(argv, cwd=tmp_path)
    assert result.returncode == 0
    assert not custom_log.exists()
    assert not (tmp_path / "csubst.log").exists()


@pytest.mark.slow
def test_download_no_download_missing_resource_fails_cleanly(tmp_path):
    cache_dir = tmp_path / "empty-cache"
    result = _run_csubst(
        ["download", "--resource", "vesm-35m", "--resource_cache_dir", str(cache_dir), "--no_download", "yes"],
        cwd=tmp_path,
    )
    assert result.returncode == 2
    assert "automatic download is disabled" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("command", "outdir"),
    [
        ("search", "csubst_search"),
        ("analyze", "csubst_search"),
        ("inspect", "csubst_inspect"),
        ("benchmark", "csubst_benchmark"),
        ("benchmark-plot", "csubst_benchmark_plot"),
        ("doctor", "csubst_doctor"),
        ("simulate", "csubst_simulate"),
        ("sites", "csubst_sites"),
    ],
)
def test_subcommand_output_namespace_defaults_are_command_specific(command, outdir):
    from csubst import cli

    parser = cli._build_parser()
    parsed = parser.parse_args([command])
    assert parsed.outdir == outdir
    assert parsed.output_prefix == "csubst"
    if command == "inspect":
        assert parsed.combination_count_max_arity == 10


def test_advanced_options_remain_parseable_in_normal_execution_mode():
    from csubst import cli

    parser = cli._build_parser()

    sites = parser.parse_args(
        ["sites", "--expected_state_backend", "eigen", "--database_timeout", "45"]
    )
    assert sites.expected_state_backend == "eigen"
    assert sites.database_timeout == 45

    search = parser.parse_args(["search", "--epistasis_beta", "auto"])
    assert search.epistasis_beta == "auto"


def test_independent_context_cannot_be_overwritten_by_log(tmp_path):
    context = tmp_path / 'context.tsv'
    original = 'branch_key\tcontext_1\n'
    context.write_text(original)
    result = _run_csubst(['search', '--epistasis_context_file', str(context),
                         '--log_file', str(context)], cwd=tmp_path)
    assert result.returncode != 0
    assert context.read_text() == original
