# M2: local lecture-notes agent

## Scope and implementation

The MVP accepts audio/video files (audio track), produces timestamped ASR, and
uses a genuine model-directed notes loop. The local model selects
`read_transcript(start,end)`, `search_transcript(query)`, `save_notes(markdown)`
and `finish()`. It can revisit source material or revise a draft. Tool execution,
validation, limits and trace serialization are deterministic. ASR preparation is
a fixed preprocessing stage, not falsely presented as a model decision.

Short-term memory is the visible conversation/tool history and current saved
draft. Long-term course retrieval, speaker diarization, slide understanding,
PDF ingestion and web search were deferred. This is an explicit reduction of
M1, not an implementation of its full multimodal architecture.

### Tested local setup

- NVIDIA A100-SXM4-80GB; Linux; Python 3.12.3.
- Ollama 0.40.11, `qwen3:4b`, **4.0B, Q4_K_M**, about 2.5 GB weights.
  At evaluation time the tag resolved to **Qwen3-4B-Thinking-2507**.
  Thinking output disabled; temperature 0, seed 42, 16,384 context,
  maximum 3,000 output tokens. The tag is mutable: use the stored digest and
  metadata in [runtime.json](../eval/results/runtime.json) to identify the run.
- faster-whisper 1.2.1, `small.en`, CTranslate2 4.8.4; full recording ASR used
  GPU float16, batch 16, beam 5, VAD. See [ASR provenance](../eval/asr_provenance.json).
- API model spend: $0. Hardware/electricity cost is not measured.

Smaller models were chosen instead of M1's Omni/ASR/VideoLLaMA combination to
get a reproducible, measured local loop working quickly. The tested setup is
not a minimum hardware requirement; frozen-transcript runs can use CPU Ollama.

## Reproduction

Exact installation/run commands are in [README.md](../README.md). A clean,
isolated environment was created from `uv.lock`; 20 tests passed and the CLI
entry point worked. A real 147-second media interval was transcribed on CPU
and converted to notes with the local Qwen loop, not a mock.
Its [source, ASR trace, agent trace and notes](../examples/media-smoke/) are preserved.

The full recording was transcribed in 89.88 seconds with cached model weights
on this A100, including extraction. Initial model/runtime downloads and failed
development attempts are not included in that preparation latency.

Evaluation command:

```bash
uv run lecture-notes eval --output runs/reproduction --judge
```

Never overwrite the committed sample or existing run artifacts. Trace logs
contain exact visible model contexts, structured decisions with brief summaries,
tool arguments/results, errors and final outputs. They do not request private
chain-of-thought. A saved Markdown file may be a partial draft: check run status.

## Evaluation design

The frozen sample is **20 section-level tasks from ONE recording**, not 20
independent lectures. It includes the lecture and two corrected seminar
derivations. Frozen in commit `4e5709a` before measuring the agent.

The supplied Russian conspect was used for expected concepts, **not** supplied
to the agent/baseline. Labels are coding-agent authored, not independently
human-verified. Frozen ASR preserves recognition errors. See
[eval/README.md](../eval/README.md) for provenance, methods and limitations.

Methods retained from M1:

- Word-search concept coverage, now explicitly a **lexical mention proxy**.
- A separate local LLM judgment of source support and clarity. The judge samples
  up to eight claims, uses the same model as generation, and is fallible.
- Basic trajectory/structure checks and manual trace inspection supplement
  final-answer scoring. This is not exhaustive decision accuracy evaluation.

Standalone **factual precision** against independently verified reference claims
was deferred: the current annotations and judge do not justify that label.
Faithfulness-to-ASR is not factual correctness, especially when ASR is wrong.

Baseline: one local-model call with all the same scoped source, same task, no
tools or feedback. Agent success proxy = finished + valid structure/citations +
at least 75% lexical concept coverage. It does **not** establish factual success.
Downloads, ASR, and judge calls are excluded from generation latency/calls;
judge overhead is separately reported in the machine-readable summary.

The first evaluation process was interrupted by a 120-second harness timeout.
Completed task pairs were preserved; the unfinished t05 attempt was archived
under `eval/results/interrupted/` and rerun. The remaining run used `--resume`
with identical prompts/model settings. Archived abandoned calls are not included
in completed-task means. A media smoke test briefly shared the inference server;
latencies are first operational measurements, not isolated hardware benchmarks.

## Results

Measured summary and interpretation are finalized after all 20 paired runs.
Raw artifacts are in [eval/results](../eval/results).

### Judge audit: high scores are not proof

The judge repeats t01's misplaced definition timestamp and marks t03's unsupported
variance/decomposition link as supported. In t04 it even accepts a confusion
between integrating `P(y|x)` over `y` and integrating `P(x|y)` over `y`.
These are concrete false-positive/attribution problems in the evaluator itself,
not just hypothetical uncertainty. Inspect the separate judge traces and source;
do not read a high mean faithfulness proxy as an independently verified precision.

For a small independent arithmetic check of the supplied reference assumptions:

| Worked example | Recomputed posterior |
|---|---:|
| Alarm; burglary prior 0.0002, earthquake prior 0.01, false-alarm rate 0.1 | 0.166694 (about 17%) |
| Same alarm model, burglary prior 0.002 | 0.667111 (about 67%) |
| Alarm plus radio announcement of earthquake, burglary prior 0.002 | 0.019646 (about 2%) |
| Disease prior 0.0001, sensitivity/specificity both 0.99, positive test | 0.009804 (about 1%) |

These checks do not resolve every corrupted ASR number, certify the model's
assumptions, or constitute a full independent annotation audit. In particular,
the ASR's “70%” near the initial alarm posterior conflicts with its nearby “17%”
and the conspect calculation. Treat that passage as uncertain, not as a reliable
number to copy merely because a source-grounding judge accepts it.

## Three annotated failures

These are actual local-model runs, not constructed mock traces. JSONL is
one-based: each event's `seq` is also its line number. For M3, freeze the model
request immediately before the cited decision, stub the recorded tools, and
test a better next action without editing the historical trace.

### 1. t01: plausible notes, misplaced source citations

- [Notes](../eval/results/agent/t01.md), [trace](../eval/results/agent/t01.trace.jsonl).
- **Step 3, seq 14–16:** `save_notes` attributes the conditional-distribution
  definition to `[00:03:04–00:03:35]`. The source in step 1 / seq 6 only introduces
  the topic there; the actual ratio definition is in `[00:03:35–00:04:12]`.
- **Step 4, seq 19:** `finish` claims citations match the source, based on
  no-issue structural feedback. The validator checks syntax/scope, not entailment.
- **Likely cause:** coarse source attribution and overtrust in a structural
  check. A mathematically correct definition can still have a wrong citation.
- **Next fix:** validate/ask the model to match each claim to a specific source
  segment; distinguish “citation format valid” from “claim supported here.”
- **M3 decision test:** after the save observation, prefer rechecking the ratio
  passage before confidently finishing.

### 2. t02: ineffective repair loop reaches the step limit

- [Draft](../eval/results/agent/t02.md), [trace](../eval/results/agent/t02.trace.jsonl),
  [status](../eval/results/agent/t02.stats.json).
- **Step 2, seq 9–11:** `save_notes` uses backtick timestamp ranges instead of
  bracket citations; the tool reports missing citations.
- **Step 3, seq 14–16:** the model says it will fix the citation format but saves
  the same draft. Steps 4–10 repeat this. The run ends with `step_limit`, not
  success, after 10 model calls; the draft remains available for diagnosis.
- **Likely cause:** instruction-following/copying failure plus feedback that
  tells it the rule but not the specific minimal edit. Repeating the entire
  draft in visible history can reinforce the bad format.
- **Next fix:** actionable feedback with a literal correct citation example,
  explicit identical-draft detection, and a tested policy for repeated nonprogress.
- **M3 decision test:** freeze the prefix before step 3; require a real citation
  change, not another identical `save_notes` claiming to repair it.

### 3. t03: unsupported connection between two neighboring topics

- [Notes](../eval/results/agent/t03.md), [trace](../eval/results/agent/t03.trace.jsonl).
- **Step 2, seq 9–11:** `save_notes` adds “This decomposition is useful for
  reducing variance.” Source step 1 / seq 6 discusses variance reduction from
  *observing another variable*, followed by the algebraic product rule. It does
  not establish that factorizing a joint distribution itself reduces variance.
- **Step 3, seq 14:** `finish` asserts the notes add no external knowledge;
  structurally valid timestamps do not expose this source-conflation error.
- **Likely cause:** merging adjacent lecture topics into a plausible causal
  statement, with insufficient semantic review of the final claim.
- **Next fix:** remove the unsupported link or distinguish the two ideas in
  separate bullets, with claim-to-segment review rather than general plausibility.
- **M3 decision test:** given the saved draft and source, revise that bullet
  instead of declaring it grounded.

## Updated M1 failure map

| Predicted failure | What we actually observed |
|---|---|
| Technical/math term recognition | Real ASR contains “product row,” “livelihood,” and inconsistent spoken-number recognition. A glossary may help spelling but cannot establish missing math. |
| Losing lecture information / omitting details | Section-level evaluation limits context, but lexical recall cannot establish complete semantic coverage. No claim of whole-lecture recall. |
| Student versus lecturer confusion | Seminar includes incorrect proposals followed by corrections; no diarization is implemented. Coverage alone cannot prove that the final corrected statement was selected. |
| Important visual-only information | Not measured by this audio-only MVP. No claim of slide/formula recovery. |
| Bad visuals / invented formulas | Visual quality itself is untested. Unsupported source connections and coarse citations were observed (t01/t03). |
| Agent omits important details | Remains a risk; imperfect aliases can miss or falsely count concepts. Human audit is required. |
| New: repair stagnation | Observed repeated unchanged saves after feedback (t02); bounded execution prevents an infinite loop but does not fix the notes. |

## Remaining submission work

The team must record/upload the narrated local-model demo and replace the pending
README link. See [recording guide](RECORDING.md). Work stays on `first-iteration`
as requested, but the rubric requires timely integration into `main` for grading.
Repository push permissions also need to be available; local commits alone are
not a completed remote submission.
