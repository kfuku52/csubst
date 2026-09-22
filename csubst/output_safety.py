"""Shared destination checks and finalization for one CLI invocation."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, TypedDict
from urllib.parse import quote


class _OutputState(TypedDict):
    protected: tuple[tuple[str, str], ...]
    finalizers: dict[str, Callable[[], None]]


_active: ContextVar[_OutputState | None] = ContextVar('csubst_output_safety', default=None)


def same_path(left, right):
    if os.path.realpath(left) == os.path.realpath(right):
        return True
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, NotADirectoryError):
        return False


def check_destination(path, protected):
    for label, source in protected:
        if same_path(path, source):
            raise ValueError('Output must not overwrite {}: {}'.format(label, source))


def validate_destination(path):
    state = _active.get()
    if state is not None and isinstance(path, (str, os.PathLike)):
        check_destination(path, state['protected'])


def register_finalizer(path, callback):
    state = _active.get()
    if state is not None:
        state['finalizers'][os.path.abspath(path)] = callback


@contextmanager
def output_context(protected):
    state: _OutputState = {'protected': tuple(protected), 'finalizers': {}}
    token = _active.set(state)
    try:
        yield
    finally:
        try:
            for finalize in tuple(state['finalizers'].values()):
                finalize()
        finally:
            _active.reset(token)


def trait_filename(trait):
    """Encode labels injectively while retaining historical simple filenames."""
    return quote(str(trait), safe='._-')
