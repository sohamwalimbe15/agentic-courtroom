from types import SimpleNamespace as NS

import pytest

from courtroom import llm

PASSAGE = "The old lighthouse keeper never spoke of his brother.  He kept the lamp lit every night."


class FakeClient:
    """Judges (max_tokens 60) answer with JSON, everyone else quotes one real and one fake line."""

    def __init__(self):
        self.calls = 0
        self.messages = NS(create=self.create)

    async def create(self, **kw):
        self.calls += 1
        if kw["max_tokens"] == 60:
            text = '{"A": 0.8, "B": 0.2}'
        else:
            text = (
                "He is right. <quote>never spoke of his brother</quote> and "
                "<quote>he sold the boat</quote>"
            )
        usage = NS(
            input_tokens=1000,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return NS(content=[NS(type="text", text=text)], usage=usage, stop_reason="end_turn")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    from courtroom import app, run

    for mod in (run, app):  # defaults depend on which API keys the shell has
        monkeypatch.setattr(mod, "DEBATER", "claude-sonnet-5-5")
        monkeypatch.setattr(mod, "JUDGE", "claude-haiku-5-5")
    c = FakeClient()
    llm.configure(5.0, 4, client=c)
    return c


@pytest.fixture
def question():
    return {
        "id": "q1",
        "split": "test",
        "title": "T",
        "passage": PASSAGE,
        "question": "What did the keeper avoid?",
        "correct": "his brother",
        "wrong": "the sea",
    }
