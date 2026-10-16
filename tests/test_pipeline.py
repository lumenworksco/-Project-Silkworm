import json

import generator
import topics
import uploader


def test_sample_prompt_contains_topic():
    all_topics = {t for ts in topics.TOPICS.values() for t in ts}
    for _ in range(50):
        topic, prompt = topics.sample_prompt()
        assert topic in all_topics
        assert topic in prompt


def test_templates_format_cleanly():
    for template in topics.PROMPT_TEMPLATES:
        assert "{topic}" in template
        assert "{" not in template.format(topic="x")


def _fake_chat(helpful, terse):
    replies = iter([helpful, terse])
    return lambda messages, max_tokens: next(replies)


def _run(monkeypatch, helpful, terse):
    monkeypatch.setattr(generator, "_chat", _fake_chat(helpful, terse))
    monkeypatch.setattr(generator.time, "sleep", lambda s: None)
    return generator.generate_responses("q")


def test_accepts_good_pair(monkeypatch):
    good = "a" * generator.MIN_CHOSEN_LENGTH
    terse = "b" * generator.MIN_REJECTED_LENGTH
    assert _run(monkeypatch, good, terse) == (good, terse)


def test_rejects_short_chosen(monkeypatch):
    assert _run(monkeypatch, "a" * (generator.MIN_CHOSEN_LENGTH - 1), "b" * 40) is None


def test_rejects_short_rejected(monkeypatch):
    assert _run(monkeypatch, "a" * 300, "b" * (generator.MIN_REJECTED_LENGTH - 1)) is None


def test_rejects_identical(monkeypatch, ):
    monkeypatch.setattr(generator, "MIN_CHOSEN_LENGTH", 10)
    assert _run(monkeypatch, "same text here", "same text here") is None


def test_rejects_failed_call(monkeypatch):
    assert _run(monkeypatch, None, "b" * 40) is None


def test_consolidate_merges_and_clears_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(uploader, "DATA_DIR", tmp_path)
    monkeypatch.setattr(uploader, "PENDING_PATH", tmp_path / "pending.jsonl")
    monkeypatch.setattr(uploader, "LOCAL_DATA_PATH", tmp_path / "dataset.parquet")
    row = {"prompt": "p", "chosen": "c", "rejected": "r", "topic": "t", "timestamp": "now"}
    uploader.save_example(row)
    uploader.save_example({**row, "prompt": "p2"})
    assert uploader.consolidate() == 2
    assert not (tmp_path / "pending.jsonl").exists()
    uploader.save_example({**row, "prompt": "p3"})
    assert uploader.consolidate() == 1
    assert uploader.load_seen_prompts() == {"p", "p2", "p3"}


def test_save_stats_omits_private_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(uploader, "STATS_PATH", tmp_path / "stats.json")
    uploader.save_stats({"total_accepted": 3, "_start_ts": 1.0})
    assert json.loads((tmp_path / "stats.json").read_text()) == {"total_accepted": 3}
