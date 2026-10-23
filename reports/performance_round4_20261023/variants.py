"""Isolated proposals applied to the immutable c371a1b source snapshot."""

import inspect
import textwrap


SELECTION_METHOD = '''    def select_events(self, events):
        # One fixed branch axis per assignment; retain only the last size's
        # bounds check, never an event vector or a per-candidate index cache.
        size = len(events.values)
        selection = self._event_selection
        if selection is None or selection[0] != size:
            ids = self.branch_ids
            selection = (size, bool(((ids >= 0) & (ids < size)).all()))
            self._event_selection = selection
        if selection[1]:
            return events.values[self.branch_ids]
        return events.select(self.branch_ids)

'''


def replace(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def endpoint_source(source):
    return replace(source, '''                        for target, lvalues, rvalues, reducer in (
                                (reduced, left, right, observed_transform),
                                (reduced_pred, post[c, parent], np.ones_like(right),
                                 predictive_transform or observed_transform)):
                            if target is reduced_pred and not predictive:
                                continue
''', '''                        reductions = [(reduced, left, right, observed_transform)]
                        if predictive:
                            reductions.append((reduced_pred, post[c, parent], np.ones_like(right),
                                               predictive_transform or observed_transform))
                        for target, lvalues, rvalues, reducer in reductions:
''')


def diagnostic_source(source):
    source = replace(source, '''    if rate_exposure != "endpoint":
        missing_exposure = np.zeros(branch_ids.shape[0], dtype=bool)
        exposure_reasons = np.full(branch_ids.shape[0], "ok", dtype="U24")
''', '')
    source = replace(source, '''    if rate_exposure != "endpoint":
        exposure_reasons[exposure == 0] = "legacy_zero_exposure"
''', '')
    return replace(source, '    target_rate = target_event / target_exposure if target_exposure > 0 else np.nan\n',
                   '''    if rate_exposure != "endpoint":
        missing_exposure = np.zeros(branch_ids.shape[0], dtype=bool)
        exposure_reasons = np.full(branch_ids.shape[0], "ok", dtype="U24")
        exposure_reasons[exposure == 0] = "legacy_zero_exposure"
    target_rate = target_event / target_exposure if target_exposure > 0 else np.nan
''')


def selection_source(source):
    source = replace(source, '        self._parent_ids = None\n',
                     '        self._parent_ids = None\n        self._event_selection = None\n')
    source = replace(source, '    def get_lengths(self, rate_length):\n',
                     SELECTION_METHOD + '    def get_lengths(self, rate_length):\n')
    return replace(source, '        event_values = candidate_events.select(branch_ids)\n',
                   '        event_values = branch_data.select_events(candidate_events)\n')


def scalar_source(source):
    source = replace(source, '    if np.isfinite(score) and score >= 0:\n',
                     '    if math.isfinite(score) and score >= 0:\n')
    return replace(source, '    if np.isnan(score) and all(key in row for key in required):\n',
                   '    if math.isnan(score) and all(key in row for key in required):\n')


def compile_variant(function, transform):
    source = textwrap.dedent(transform(inspect.getsource(function)))
    namespace = dict(function.__globals__)
    exec(compile(source, '<round4-prototype>', 'exec'), namespace)
    return namespace[function.__name__]
