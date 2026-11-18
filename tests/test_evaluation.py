import json

import pytest

from lecture_notes.evaluation import evaluate
from lecture_notes.model import ModelError
from lecture_notes.transcript import fingerprint
from test_agent import ScriptedModel, action


def make_sample(tmp_path):
    transcript = tmp_path / "source.json"
    transcript.write_text(json.dumps({"segments": [{"start": 0, "end": 20,
        "text": "A prior represents uncertainty before data."}]}))
    sample = tmp_path / "tasks.json"
    sample.write_text(json.dumps({"transcript": "source.json", "transcript_sha256": fingerprint(transcript),
        "tasks": [{"id": "t01", "task": "Make notes", "start": 0, "end": 20,
                   "concepts": [{"name": "prior", "aliases": ["prior"]}]}]}))
    return sample


def script():
    notes = "# Prior distribution\n- The prior represents uncertainty before data. It is our initial distribution. [00:00:00]"
    return ScriptedModel([action("read_transcript", start=0, end=20),
                          action("save_notes", markdown=notes), action("finish"), {"markdown": notes}])


def test_evaluation_and_completed_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("lecture_notes.evaluation.subprocess.check_output", lambda *a, **kw: "Test GPU")
    sample = make_sample(tmp_path)
    output = tmp_path / "results"
    first = evaluate(sample, output, script())
    assert first["tasks"] == 1
    assert first["variants"]["m2_loop"]["task_success_proxy"] == 1
    second = evaluate(sample, output, ScriptedModel([]), resume=True)
    assert first["variants"]["m2_loop"] == second["variants"]["m2_loop"]
    with pytest.raises(ValueError, match="configuration"):
        evaluate(sample, output, ScriptedModel([]), judge=True, resume=True)


def test_transcript_hash_checked_before_running(tmp_path):
    sample = make_sample(tmp_path)
    (tmp_path / "source.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate(sample, tmp_path / "results", ScriptedModel([]))


def test_partial_resume_preserves_abandoned_trace(tmp_path, monkeypatch):
    monkeypatch.setattr("lecture_notes.evaluation.subprocess.check_output", lambda *a, **kw: "Test GPU")
    sample = make_sample(tmp_path)
    output = tmp_path / "results"
    partial = output / "m2_loop" / "t01.trace.jsonl"
    partial.parent.mkdir(parents=True)
    partial.write_text('{"event":"model_request","interrupted":true}\n')
    evaluate(sample, output, script(), resume=True)
    saved = output / "interrupted" / "m2_loop" / partial.name
    assert '"interrupted":true' in saved.read_text()
    assert (output / "summary.json").exists()


def test_resume_accepts_failed_tasks_without_judge_output(tmp_path, monkeypatch):
    monkeypatch.setattr("lecture_notes.evaluation.subprocess.check_output", lambda *a, **kw: "Test GPU")
    sample = make_sample(tmp_path)
    output = tmp_path / "results"
    model = ScriptedModel([ModelError("offline"), ModelError("offline"), ModelError("offline")])
    evaluate(sample, output, model, judge=True)
    summary = evaluate(sample, output, ScriptedModel([]), judge=True, resume=True)
    assert summary["variants"]["m2_loop"]["finished"] == 0
    assert summary["variants"]["m2_loop"]["judged_tasks"] == 0


def test_all_variants_share_evaluator_and_trace_identity(tmp_path, monkeypatch):
    monkeypatch.setattr("lecture_notes.evaluation.subprocess.check_output", lambda *a, **kw: "Test GPU")
    sample = make_sample(tmp_path)
    notes = "# Prior distribution\n- The prior represents uncertainty before data. " \
            "It is our initial distribution. [00:00:00]"
    model = ScriptedModel([
        action("read_transcript", start=0, end=20), action("save_notes", markdown=notes), action("finish"),
        action("read_transcript", start=0, end=20), action("save_notes", markdown=notes), action("finish"),
        action("read_transcript", start=0, end=20), action("save_notes", markdown=notes),
        action("check_claims"), action("finish"),
        {"markdown": notes},
    ])
    variants = ("m2_loop", "lean_loop", "claim_check", "one_call_baseline")
    summary = evaluate(sample, tmp_path / "results", model, variants=variants)
    assert set(summary["variants"]) == set(variants)
    for variant in variants:
        trace = tmp_path / "results" / variant / "t01.trace.jsonl"
        assert all(row["variant"] == variant for row in map(json.loads, trace.read_text().splitlines()))
