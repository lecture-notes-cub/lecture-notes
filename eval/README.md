# Frozen M2 evaluation sample

**20 section-level study-note tasks, from one user-supplied recording.** This is
not a sample of 20 independent lectures. Sections span elementary definitions,
statistical estimation, intuition/examples, causal modeling, and two seminar
problems with student mistakes and corrections.

- `tasks.json`: frozen prompts, intervals, reference locations, and expected
  concepts/lexical aliases; frozen in commit `4e5709a` before evaluated outputs.
- `lecture_excerpt.json`: exact ASR segments available to the evaluated tools,
  preserving their original recording timestamps and recognition errors.
- `asr_provenance.json`: hashes, local ASR configuration and one-time preparation
  latency. Full original media/PDF are not redistributed.
- `results/`: all 20 agent runs, 20 baseline runs, separate judge traces, raw
  per-task metrics, aggregate summary and inference runtime/model metadata.

## Provenance and annotations

Lecture: https://www.youtube.com/watch?v=XwR767IfQHI . The 3h21m recording contains
about 90 minutes of lecture plus a seminar. The Russian reference conspect is
linked in [THIRD_PARTY.md](../THIRD_PARTY.md). Its PDF pages 4–10 cover Lecture 1;
the Gamma-Poisson exercise also relates to pages 11–12. The variance/coin/elephant
examples and seminar details were additionally checked against the ASR discussion.

Task boundaries were snapped to real segment starts, with occasional overlap
for context. The excerpt is the union of these intervals. A task can only access
segments whose start lies in its own interval. Last segments sometimes end a
fraction after the next boundary; this is ASR segmentation, not word alignment.

**Labels were authored by the coding agent from the supplied conspect and ASR,
not independently verified by a human or by listening to all the audio.** They
are a practical initial fixture, not trusted semantic gold. The team should
review them before making strong quality claims. Do not tune aliases after
seeing outputs and then describe that as a held-out result.

The agent and baseline receive task wording, interval and ASR only. Neither gets
concept labels, reference notes, the other variant's output, or judge feedback.

## Methods and levels

1. **Lexical concept-mention coverage (task level):** the fraction of four
   expected concepts with at least one normalized, word-bounded alias in notes.
   Cheap and deterministic, following M1's proposed word-search approach. It
   checks mention, **not whether the relation, formula, or number is correct**.
   Numeric aliases and broad words can be permissive; paraphrases can be missed.
2. **Local grounding/clarity judge (task level):** same Qwen model, separate call,
   source plus final notes, up to eight substantive claims. Faithfulness proxy
   is the fraction marked supported. Clarity is rated 1–5. Same-model bias,
   missed claims, ASR errors and imperfect judging make these uncertain proxies.
3. **Structure and trajectory checks:** headings, nonempty useful-length notes,
   timestamp format/scope; finish status, source reads, model/tool calls, errors.
   Trace inspection supports a first decision-level failure analysis. These
   checks do not certify source entailment or exhaustively score planning.

**Task-success proxy:** model finished, structure checks passed, and lexical
coverage ≥75%. This threshold was defined before the measured run. It is not
a factual-success rate; the judge is reported separately, not used to tune it.

Baseline: one local model call with the complete same scoped transcript and the
same task, requesting Markdown with citations. No tools, revision or feedback.
Both variants use the same decoding options; the baseline gets direct source
access without paying a separate read call. This is an intentionally strong,
simple Stage 0 comparison for short source intervals.

API spend is $0. Local hardware/electricity are **not measured or assumed free**.
Per-task latency/steps/calls cover notes generation, excluding downloads, ASR
preparation and optional judge calls. Judge calls/latency are reported separately.
One-time ASR latency is recorded in `asr_provenance.json`.

## Re-run / preserve for M3

```bash
uv run lecture-notes eval --output runs/new-eval --judge
```

The transcript SHA-256 is checked before evaluation; output directories must be
new. `summary.json` also records the sample hash. Model tags can drift: compare
the stored digest and model metadata in `results/runtime.json` before comparing
variants. Temperature 0 and seed 42 improve repeatability but do not promise
bit-identical GPU outputs across software/hardware versions.

For an interrupted run, repeat the same command with `--resume`. Configuration
and completed trace identities are checked; incomplete attempts are preserved
under `interrupted/` and rerun, not silently overwritten or counted as successes.

Do not regenerate the frozen transcript for M3: fresh ASR can change recognition
and segmentation. `scripts/freeze_sample.py` documents the initial selection;
it refuses to overwrite existing fixtures. To test ASR changes, create a separate
versioned evaluation condition rather than silently replacing this environment.
