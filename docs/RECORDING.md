# Required local-model demo (3–5 minutes)

The team must record this with a voiceover and upload it to an accessible URL.
Replace the pending recording link in README.md. Do not claim the video exists
until it has actually been uploaded and access checked.

## Preparation

1. Start Ollama (`ollama serve`, if not already running).
2. Run `ollama list`; show `qwen3:4b`.
3. Use a new output directory (the program refuses to overwrite old traces).
4. Have `eval/tasks.json`, `docs/M2_REPORT.md`, and two terminal panes visible.

## Suggested voiceover and commands

**0:00–0:40 — purpose and honest scope.**
“We turn lecture audio into timestamped Markdown study notes using local models.
Whisper transcribes audio; Qwen chooses transcript tools, writes/reviews notes,
and decides to stop. We do not yet analyze slides or identify speakers.”

**0:40–2:40 — run three frozen tasks.**

```bash
uv run lecture-notes eval --limit 3 --output runs/recording-01
```

While it runs, in the second pane:

```bash
tail -F runs/recording-01/agent/t01.trace.jsonl
```

Show the file being appended to, plus `step N: read_transcript`, `save_notes`,
and the resulting Markdown. Explain that the action/reason/arguments come from
the model, not a hard-coded sequence. Show tasks 2 and 3 in the same way; use
the corresponding `t02.trace.jsonl` and `t03.trace.jsonl` files.

**2:40–3:40 — metrics and cost.**
Open the committed `eval/results/summary.json` and compare agent vs baseline.
Explain lexical concept coverage, finish rate, calls, and latency. API spend is
$0, but GPU/electricity are not free or measured. A same-model judge is fallible.

**3:40–4:40 — one real failure and next improvement.**
Open one annotated failure in the M2 report and its exact trace decision.
Explain the source evidence, the model's mistake, and a plausible fix.
End by stating the limited diversity: 20 section-level tasks from one lecture,
not 20 independent lectures or a general quality guarantee.

Record actual execution, not an edited/replayed trace presented as a live run.
Cold model loading/CPU inference can take longer; warm up before recording and
say that setup/ASR time is excluded from the frozen-transcript demo.
