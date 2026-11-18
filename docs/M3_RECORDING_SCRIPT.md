# M3 recording script (3–5 minutes)

This recording demonstrates `m2_loop` and `lean_loop` on the same frozen task while
their traces are written. Use fresh output paths: the program intentionally refuses
to overwrite an existing run. Warm the model before recording, but run both demos
live rather than presenting saved traces as live execution.

## Before recording

Arrange three terminal panes and keep `docs/M3_REPORT.md` ready in an editor. Start Ollama and
confirm the tested local model:

```bash
ollama serve
ollama list
```

The expected model is Qwen3 4B, `qwen3:4b`, Q4_K_M quantization. The measured M3
runs used Ollama 0.40.16, temperature 0, seed 42 and a 16,384-token context on an
NVIDIA A100 80 GB. Inference is local, so API cost is $0; hardware and electricity
are real costs but were not measured.

Warm up outside the recording if necessary:

```bash
uv run lecture-notes run --variant m2_loop \
  --transcript eval/lecture_excerpt.json --start 184.35 --end 331.6 \
  --task "Make concise, timestamped study notes on conditional probability and independence." \
  --output runs/m3-warmup/notes.md
```

## Timed voiceover and actions

### 0:00–0:35 — task and variants

Show the variant definitions in `docs/M3_REPORT.md` and say:

> This agent turns a lecture transcript into timestamped Markdown study notes. M3
> compares the M2 ReAct-style loop with two mechanism changes on the same frozen
> 20-task sample. The lean variant hides search, allows one complete read and at
> most two saves, and stops after four decisions. The claim-check variant enforces
> a source-grounding check before finish. These hypotheses were committed before
> the measured evaluation.

### 0:35–1:30 — run `m2_loop` and show its live trace

In pane 1, run the M2 loop on t01's exact interval:

```bash
uv run lecture-notes run --variant m2_loop \
  --transcript eval/lecture_excerpt.json --start 184.35 --end 331.6 \
  --task "Make concise, timestamped study notes on conditional probability and independence. Preserve key definitions, reasoning and examples from this interval; flag unclear speech." \
  --output runs/m3-video-m2/notes.md
```

As soon as the trace exists, use pane 2:

```bash
tail -F runs/m3-video-m2/notes.trace.jsonl
```

Say while it runs:

> Each JSONL record is flushed as it happens and includes the variant name. The
> model selects actions and arguments; code validates them and bounds the loop.
> Here the M2 agent can read a passage, use literal search, save or revise a draft,
> and decide when to finish. The trace stores exact visible requests and responses,
> tool observations, token counts and status, but asks only for a short operational
> reason rather than private chain-of-thought.

When it completes, briefly show `runs/m3-video-m2/notes.md` and its timestamped
citations.

### 1:30–2:25 — run `lean_loop` on the same task

Stop the first `tail` with Ctrl-C. In pane 1:

```bash
uv run lecture-notes run --variant lean_loop \
  --transcript eval/lecture_excerpt.json --start 184.35 --end 331.6 \
  --task "Make concise, timestamped study notes on conditional probability and independence. Preserve key definitions, reasoning and examples from this interval; flag unclear speech." \
  --output runs/m3-video-lean/notes.md
```

In pane 2:

```bash
tail -F runs/m3-video-lean/notes.trace.jsonl
```

Say:

> This is the same model, task, source and decoding setup; only the agent mechanism
> changes. Lean exposes no search tool, requires one read of the whole interval,
> permits at most two saves and caps the run at four decisions. Its predicted trace
> is read, draft, optional revision, finish. This makes the comparison causal and
> reproducible rather than a model swap or prompt-only variant.

Show `runs/m3-video-lean/notes.md` and, if time permits, both `.stats.json` files.

### 2:25–3:25 — measured comparison and costs

Open the aggregate table in `docs/M3_REPORT.md` and say:

> On the measured 20 tasks, M2 finished and produced valid structure on all 20 and
> passed the preregistered M3 success proxy on 18, or 90%. Lean passed 15, or 75%,
> while using 24% fewer calls and 23% fewer tokens. Claim-check passed only 4 and
> used about 5.5 times the M2 tokens because its enforced re-check loop encouraged
> repeated revisions. The one-call baseline passed 16 and was cheapest at one call,
> 1,681 tokens and 5.48 seconds per task. Local API spend was zero for every variant.

> Success is not just finish. It requires finish, valid structure, and either every
> human-verified key-claim check on a labeled task or at least 75% lexical coverage
> elsewhere. Lexical coverage is only a continuity proxy: raw transcript copying
> scored 97.5% at M2. Dzhamilia manually checked every key-claim description,
> numerical value and evidence interval against the frozen transcript before these
> runs; the labels were never shown to the agents. The same local judge agreed with binary key-claim outcomes on
> 10 of 16 labeled variant-task pairs, so neither proxy is treated as ground truth.

### 3:25–4:15 — repeats, failure, and rational winner

Show the pass³ table and one claim-check trace from `eval/m3_results/`, then say:

> We repeated every configuration three times. M2 achieved strict pass-cubed on 17
> of 20 tasks and pass-at-three on 19. Claim-check ranged from 20% to 50% success,
> but only one task passed all three runs. A typical failure is save, rejected early
> finish, check, revision, then repeated invalidation and re-checking until the step
> cap. This directly rejects our hypothesis that an enforced self-check would improve
> grounded quality efficiently.

> We select `m2_loop` because our PEAS performance measure prioritizes useful,
> grounded, structurally valid notes, then fewer calls, tokens and seconds. It has
> the highest success, full completion, highest judge faithfulness and highest
> pass-cubed. If throughput were primary, the one-call baseline would be rational
> instead. These are correlated sections from one lecture, so full-lecture and
> cross-domain performance remain untested.

End by showing that both new trace files exist. Keep the final recording between
three and five minutes, upload it to an accessible URL, test access in a private
window, and replace or supplement the README video link only after upload.
