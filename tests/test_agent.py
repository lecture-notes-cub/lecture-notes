import json

import pytest

from lecture_notes.agent import Tools, action_schema, run, validate_notes
from lecture_notes.evaluation import coverage, key_claim_score
from lecture_notes.model import ModelError
from lecture_notes.transcript import load_transcript, timestamp


@pytest.fixture
def transcript(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"segments": [
        {"start": 10, "end": 20, "text": "The prior encodes uncertainty before data."},
        {"start": 21, "end": 30, "text": "Bayes updates the prior to the posterior."},
    ]}))
    return path


class ScriptedModel:
    model = "test-stub-not-a-real-evaluation"

    def __init__(self, actions):
        self.actions = iter(actions)

    def complete(self, messages, schema, trace, step):
        trace.emit("model_request", step=step, messages=messages)
        try:
            action = next(self.actions)
        except StopIteration as exc:
            raise AssertionError(f"ScriptedModel exhausted at step {step}") from exc
        if isinstance(action, Exception):
            raise action
        trace.emit("model_response", step=step, response=action)
        return action


def action(name, **kwargs):
    return {"reason": "Test decision", "action": name, "arguments": kwargs}


def test_model_selects_tools_recovers_and_stops(transcript, tmp_path):
    model = ScriptedModel([
        action("read_transcript", start=-1, end=30),
        action("read_transcript", start=10, end=30),
        action("search_transcript", query="missing"),
        action("save_notes", markdown="# Bayesian update\n- The prior encodes uncertainty before data. "
               "After observations it becomes the posterior. [00:00:10]"),
        action("finish"),
    ])
    output = tmp_path / "notes.md"
    result = run(transcript, "Summarize", output, model, 10, 30)
    assert result["status"] == "finished"
    assert result["model_calls"] == 5
    events = [json.loads(line) for line in output.with_suffix(".trace.jsonl").read_text().splitlines()]
    assert any(e["event"] == "observation" and "error" in e["result"] for e in events)
    assert events[-1]["event"] == "run_end"
    assert events[-2]["event"] == "final_answer"


def test_step_limit_is_not_success(transcript, tmp_path):
    result = run(transcript, "Summarize", tmp_path / "notes.md",
                 ScriptedModel([action("search_transcript", query="missing")]), 10, 30, max_steps=1)
    assert result["status"] == "step_limit"
    assert not result["has_draft"]


def test_model_timeout_recovery(transcript, tmp_path):
    model = ScriptedModel([ModelError("timeout"), action("finish")])
    result = run(transcript, "Summarize", tmp_path / "notes.md", model, 10, 30, max_steps=2)
    assert result["status"] == "step_limit"


def test_tools_validate_scope_and_finish(transcript, tmp_path):
    tools = Tools(load_transcript(transcript), 10, 30, tmp_path / "notes.md")
    for args in ({"start": float("nan"), "end": 30}, {"start": 10, "end": 31},
                 {"start": "10", "end": 30}):
        with pytest.raises(ValueError):
            tools.call("read_transcript", args)
    with pytest.raises(ValueError):
        tools.call("finish", {})
    assert tools.call("search_transcript", {"query": "PRIOR"})["total"] == 2
    assert tools.call("search_transcript", {"query": "absent"})["total"] == 0


def test_baseline_one_call(transcript, tmp_path):
    model = ScriptedModel([{"markdown": "# Notes\nThe prior represents uncertainty. [00:00:10]"}])
    result = run(transcript, "Summarize", tmp_path / "notes.md", model, 10, 30, baseline=True)
    assert result["status"] == "finished"
    assert result["model_calls"] == 1
    assert result["tool_calls"] == 0


def test_coverage_is_word_bounded():
    concepts = [{"name": "prior", "aliases": ["prior"]}]
    assert coverage("priority", concepts)["concept_coverage"] == 0
    assert coverage("The PRIOR.", concepts)["concept_coverage"] == 1


def test_key_claim_score_checks_required_groups_and_forbidden_phrases():
    labels = [{"id": "posterior", "description": "correct posterior",
               "required": [["17%", "one sixth"], ["posterior"]], "forbidden": ["70% probability"]}]
    assert key_claim_score("The posterior is one sixth.", labels)["key_claim_score"] == 1
    assert key_claim_score("The posterior is 70% probability.", labels)["key_claim_score"] == 0


def test_lean_loop_hides_search_and_enforces_bounded_plan(transcript, tmp_path):
    assert "search_transcript" not in action_schema("lean_loop")["properties"]["action"]["enum"]
    tools = Tools(load_transcript(transcript), 10, 30, tmp_path / "notes.md", "lean_loop")
    tools.call("read_transcript", {"start": 10, "end": 30})
    with pytest.raises(ValueError, match="one read"):
        tools.call("read_transcript", {"start": 10, "end": 30})
    with pytest.raises(ValueError, match="unavailable"):
        tools.call("search_transcript", {"query": "prior"})


def test_claim_check_blocks_unchecked_or_revised_draft(transcript, tmp_path):
    tools = Tools(load_transcript(transcript), 10, 30, tmp_path / "notes.md", "claim_check")
    draft = "# Notes\n" + "The prior represents uncertainty before data. " * 2 + "[00:00:10]"
    tools.call("save_notes", {"markdown": draft})
    with pytest.raises(ValueError, match="check_claims"):
        tools.call("finish", {})
    review = tools.call("check_claims", {})
    assert "source_evidence" in review
    assert tools.call("finish", {})["final_answer"] == draft
    tools.call("save_notes", {"markdown": draft + "\nRevised."})
    with pytest.raises(ValueError, match="check_claims"):
        tools.call("finish", {})


def test_variant_is_on_every_trace_record(transcript, tmp_path):
    model = ScriptedModel([action("read_transcript", start=10, end=30),
                           action("save_notes", markdown="# Notes\n" + "Grounded notes. " * 8 + "[00:00:10]"),
                           action("finish")])
    output = tmp_path / "notes.md"
    run(transcript, "Summarize", output, model, 10, 30, variant="m2_loop")
    rows = [json.loads(line) for line in output.with_suffix(".trace.jsonl").read_text().splitlines()]
    assert rows and all(row["variant"] == "m2_loop" for row in rows)


def test_citation_validation():
    assert validate_notes("# Notes\n" + "x" * 90 + " [00:00:10]", 10, 30)["issues"] == []
    assert validate_notes("# Notes\n" + "x" * 90 + " [00:99:10]", 10, 30)["issues"]
    assert not validate_notes("# Notes\n" + "x" * 90 + " [00:00:10–00:00:20]", 10, 30)["issues"]
    assert timestamp(3661) == "01:01:01"


def test_no_overwrite(transcript, tmp_path):
    output = tmp_path / "notes.md"
    output.write_text("User work")
    with pytest.raises(ValueError, match="overwrite"):
        run(transcript, "task", output, ScriptedModel([]))
    assert output.read_text() == "User work"
