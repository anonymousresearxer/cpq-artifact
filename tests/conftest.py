import pytest

from cpq.data import load


@pytest.fixture(scope="session")
def pops():
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = load(name)
        return cache[name]

    return get
