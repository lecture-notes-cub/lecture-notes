import json

import pytest

from lecture_notes.transcript import load_transcript, transcribe


@pytest.mark.parametrize("rows", [[], [{"start": -1, "end": 2, "text": "bad"}],
    [{"start": 1, "end": 1, "text": "bad"}], [{"start": 0, "end": float("inf"), "text": "bad"}],
    [{"start": 4, "end": 6, "text": "a"}, {"start": 2, "end": 3, "text": "b"}]])
def test_rejects_invalid_segments(tmp_path, rows):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"segments": rows}))
    with pytest.raises(ValueError):
        load_transcript(path)


def test_asr_failure_leaves_trace(tmp_path):
    output = tmp_path / "transcript.json"
    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "missing.wav", output)
    events = [json.loads(line) for line in output.with_suffix(".asr.trace.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "asr_error"


def test_rejects_invalid_clip_without_loading_asr(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        transcribe("missing.wav", tmp_path / "output.json", end=float("nan"))
