"""Preflight protection for CLI inputs, without importing numerical libraries."""

import argparse
import os
from collections.abc import Iterator, Sequence

from csubst import runtime


def _path_actions(parser: argparse.ArgumentParser) -> Iterator[argparse.Action]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in {id(p): p for p in action.choices.values()}.values():
                yield from _path_actions(child)
        elif isinstance(action.metavar, str) and action.metavar.startswith('PATH'):
            yield action


def validate_log_destination(
    log_path: str, parser: argparse.ArgumentParser, args: argparse.Namespace,
    argv: Sequence[str],
) -> None:
    """Reject aliases of inputs before either truncating or appending a log.

    Inspect raw path options too: argparse may stop before populating the
    namespace when an unrelated argument has an invalid value. Path metadata
    comes from the parser so new file options receive the same protection.
    """
    actions = list(_path_actions(parser))
    defaults = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            command = action.choices.get(str(getattr(args, 'subcommand', '')))
            if command is not None:
                defaults.update({item.dest: item.default for item in _path_actions(command)})
    paths: dict[str, set[str]] = {}
    for action in actions:
        value = getattr(args, action.dest, defaults.get(action.dest))
        if isinstance(value, str) and value:
            paths.setdefault(action.dest, set()).add(value)
    options = {option: action.dest for action in actions for option in action.option_strings}
    for index, token in enumerate(argv):
        option, sep, value = str(token).partition('=')
        if not option.startswith('--'):
            continue
        matches = [name for name in options if name == option or name.startswith(option)]
        if not sep:
            if index + 1 == len(argv) or str(argv[index + 1]).startswith('--'):
                continue
            value = str(argv[index + 1])
        for name in matches:
            paths.setdefault(options[name], set()).add(value)

    # IQ-TREE files are also inputs when their locations are inferred. Protect
    # both alignment prefixes, including the full-CDS structure-aware mode.
    iqtree_outdirs = paths.get('iqtree_outdir', {'csubst_iqtree'})
    alignments = paths.get('alignment_file', set()) | paths.get('full_cds_alignment_file', set())
    inferred = {suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')
                if 'infer' in paths.get('iqtree_' + suffix, {'infer'})}
    for alignment in alignments:
        for outdir in iqtree_outdirs:
            prefix = runtime.infer_iqtree_output_prefix(alignment, outdir)
            for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log'):
                dest = 'iqtree_' + suffix
                if suffix in inferred:
                    paths.setdefault(dest, set()).add(prefix + '.' + suffix)

    ignored = {'log_file', 'outdir', 'true_asr_prefix', 'epistasis_degree_outfile'}
    inferred_iqtree_inputs = {'iqtree_' + suffix for suffix in ('treefile', 'state', 'rate', 'iqtree', 'log')}
    log_realpath = os.path.realpath(log_path)
    for dest, values in paths.items():
        if dest in ignored:
            continue
        for value in values:
            if value.strip() == '':
                continue
            if value == 'infer' and dest in inferred_iqtree_inputs:
                continue
            if value == 'besthit' and dest == 'pdb':
                continue
            # Some consumers trim surrounding whitespace; protect both the
            # supplied spelling and the normalized input they will open.
            candidates = list(dict.fromkeys((value, value.strip())))
            if dest in {'prostt5_cache_file', 'sa_state_cache_file', 'vep_cache_file'}:
                base_candidates = tuple(candidates)
                candidates.extend(os.path.join(outdir, candidate)
                                  for outdir in paths.get('outdir', {'.'})
                                  for candidate in base_candidates)
            for candidate in candidates:
                same = os.path.realpath(candidate) == log_realpath
                if not same:
                    try:
                        same = os.path.samefile(candidate, log_path)
                    except (FileNotFoundError, NotADirectoryError):
                        pass
                if same:
                    raise ValueError('--log_file must not overwrite input --{}: {}'.format(dest, candidate))
