import io
from types import SimpleNamespace

import pytest

from csubst import plotting, tree


def test_text_rendering_policy_preserves_font_and_point_size():
    module = SimpleNamespace(rcParams={
        'text.hinting': 'default',
        'font.family': ['Helvetica'],
        'font.size': 5,
    })
    plotting.configure_text_rendering(module)
    assert module.rcParams == {
        'text.hinting': 'auto',
        'font.family': ['Helvetica'],
        'font.size': 5,
    }


@pytest.mark.parametrize('entry_point', ['sites', 'tree'])
@pytest.mark.parametrize('family', ['DejaVu Sans', 'Helvetica'])
def test_small_text_renders_with_auto_hinting(entry_point, family, monkeypatch):
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties, findfont

    try:
        font_path = findfont(FontProperties(family=family), fallback_to_default=False)
    except ValueError:
        pytest.skip('{} is not installed'.format(family))

    # Do not inherit an earlier test's initialization: native hinting must be
    # replaced by the actual CSUBST entry point, not by a test-only rc setting.
    monkeypatch.setattr(plotting, '_matplotlib_module', None)
    monkeypatch.setattr(plotting, '_pyplot_module', None)
    with mpl.rc_context({'text.hinting': 'default'}):
        if entry_point == 'sites':
            plotting._load_matplotlib_modules()
        else:
            tree._get_pyplot()
        assert mpl.rcParams['text.hinting'] == 'auto'

        fig = Figure(figsize=(2, 1), dpi=100)
        canvas = FigureCanvasAgg(fig)
        texts = [
            fig.text(0.1, 0.2 + index * 0.25, 'A>V Hg',
                     fontproperties=FontProperties(fname=font_path, size=size))
            for index, size in enumerate([2.5, 4, 5])
        ]
        canvas.draw()
        for text, size in zip(texts, [2.5, 4, 5]):
            assert text.get_fontsize() == size
            assert text.get_fontproperties().get_file() == font_path
            assert text.get_window_extent(canvas.get_renderer()).width > 0
        output = io.BytesIO()
        canvas.print_png(output)
        assert output.getvalue().startswith(b'\x89PNG\r\n\x1a\n')
