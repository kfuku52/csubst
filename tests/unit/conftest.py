import pytest

from factories import make_tiny_tree


@pytest.fixture
def tiny_tree():
    return make_tiny_tree()
