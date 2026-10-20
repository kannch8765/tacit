import pytest


@pytest.fixture(autouse=True)
def no_background_worker(monkeypatch):
    """Tests never spawn the detached worker."""
    monkeypatch.setenv("TACIT_NO_WORKER", "1")
