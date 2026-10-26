import json
from pathlib import Path

from lecture_notes.transcript import fingerprint, load_transcript


ROOT = Path(__file__).resolve().parents[1]


def test_twenty_frozen_tasks_and_hash():
    spec = json.loads((ROOT / "eval/tasks.json").read_text())
    transcript = ROOT / "eval" / spec["transcript"]
    assert fingerprint(transcript) == spec["transcript_sha256"]
    data = load_transcript(transcript)
    tasks = spec["tasks"]
    assert len(tasks) == 20
    assert len({t["id"] for t in tasks}) == 20
    for task in tasks:
        assert task["start"] < task["end"]
        assert len(task["concepts"]) == 4
        assert any(task["start"] <= s["start"] < task["end"] for s in data["segments"])


def test_annotated_decision_prefixes_exist():
    failures = json.loads((ROOT / "eval/failures.json").read_text())
    assert len(failures) >= 3
    for failure in failures:
        events = [json.loads(line) for line in (ROOT / "eval" / failure["trace"]).read_text().splitlines()]
        request = events[failure["request_seq"] - 1]
        decision = events[failure["decision_seq"] - 1]
        assert request["event"] == "model_request"
        assert decision["event"] == "decision"
        assert decision["step"] == failure["step"]
        assert decision["decision"]["action"] == failure["observed_action"]
