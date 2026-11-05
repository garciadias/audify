"""Library tests never call real catalogues."""

import pytest


@pytest.fixture(autouse=True)
def _no_catalogue_lookup(monkeypatch):
    monkeypatch.setenv("AUDIFY_METADATA_LOOKUP", "0")
