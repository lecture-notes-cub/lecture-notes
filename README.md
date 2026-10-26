# Lecture notes

**Group:** Dzhamilia Erkenova, Iuliia Liutova

# Task definition

### What does the agent do?

The agent transcribes video and audio recordings into notes with optional additional context.

### For whom?

Students who prefer text over video or want to recall the material.

### Why does it have to be an agent and not a script (or a one-shot ChatGPT message)?

- Multimodal data processing, which is not supported in basic ChatGPT
- Automatic text correction using context
- An agentic setup would allow for more flexibility: adding more context in the form of PDFs, adding tools such as pointing to specific time ranges in the video lecture where a specific topic is discussed, etc. This either can’t be achieved with a single script or would be hard to achieve.
- We suppose that the context of each video lecture would be too big for a one-shot prompt.
- The agent can decide to return to some part of the lecture to clarify the context.

### What model(s) are you planning to use?

- In general: local models such as Qwen3-Omni-30B-A3B-Thinking
- For audio processing: Qwen3-ASR-1.7B
- For video processing: VideoLLaMA3-7B

### Which parts are you planning to solve in code? Why?

- Multimodal interactions, if we plan on using different models for different modalities
- Specific features/tools
- Some other structural bases

We want to do this in code because this behaviour would be better off being deterministic and acting predictably.

# Initial architecture

- Perceive through the user input, video and audio, and some additional feedback from models responsible for different modalities. This is the necessary and sufficient background for the agent to be able to do the job and not be distracted.
- Memory approach:
  - Short-term: keeps the current topic, recent tool results, and a list of unresolved issues so that the notes are consistent and each next topic follows naturally from the last.
  - Long-term: has a knowledge base of different courses and lectures the user has provided before. For every course and lecture, it keeps a smaller knowledge base (transcripts, notes, the curriculum, glossary, abbreviations, theorems, etc.) so that current notes are consistent with the previous ones and a recall tool can be created/used.
  - Scratchpad of the current lecture
- We assume the following structure: Act (transcribe the lecture using a tool) -> Reason (think about what the notes should include) -> Act (write the notes) -> Observe (evaluate the notes, if necessary rewrite them, fix the grammar, etc.).

# Failure map

1. Incorrect recognition of technical/math terms -> create a predefined glossary of terms, add web search about this theme. Figured it out through group discussion.
2. The agent may not recall the whole lecture, leaving out some vital information. Figured it out through group discussion.
3. The agent may misunderstand who’s speaking and assume that a student’s phrase/question was actually the professor’s -> add mapping of text to speakers using voice detection + context. Figured it out through group discussion.
4. Important information appears only in video or audio -> sync audio and video using timestamps. Figured it out through group discussion.
5. Bad video quality could result in the agent misunderstanding the formulas/symbols, which could harm the consistency of the notes or lead to the agent hallucinating something. Figured it out through group discussion.
6. The agent omits important details. Figured it out through group discussion.

# Initial evaluation plan

- Initial/easiest evaluations:
  - Compare metrics against labeled notes (topics, formulas, theorems, etc.) on specific video lectures. Find the notes on the internet/ask the professors/students. Label a few ourselves, then maybe use an LLM to label more.

- Metrics:
  - **Concept Coverage (Recall)** - how many key ideas of the lecture remain in the notes. Use either simple word search or cosine similarity/perplexity in harder cases to check if the labeled ideas are in the agent’s notes.
  - **Factual Precision** - factual correctness. Using an LLM as a judge.
  - **Faithfulness** - the share of claims actually grounded in the lecture. Calculate similarly to recall.
  - **Note Quality** - structure, helpfulness, clarity. Using an LLM as a judge.

---

# M2 MVP: running it

This implementation follows the proposal above with a deliberately smaller scope.
**All model inference is local.** The model chooses its next tool and arguments,
can return to transcript passages, saves/revises notes, and chooses `finish`.
Code bounds execution and validates tools; it does not prescribe their order.

## Quick start (frozen lecture tasks, no audio setup required)

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and [Ollama](https://ollama.com/download). Internet is needed for initial downloads,
not subsequent inference. Allow about 8 GB free disk for the basic runtime/model;
audio/CUDA dependencies need more. CPU inference is supported but slower.

```bash
uv sync --frozen --extra dev
ollama serve
```

Leave the server running; in a **second terminal**, from this repository:

(If Ollama is already running, skip `ollama serve`.)

```bash
ollama pull qwen3:4b
uv run lecture-notes --help
uv run lecture-notes run \
  --transcript eval/lecture_excerpt.json \
  --start 184.35 --end 331.6 \
  --task "Make study notes on conditional probability and independence." \
  --output runs/demo/notes.md
```

The frozen data is committed, so this command needs no lecture download or ASR.
Each run creates:

- `notes.md`: timestamped Markdown notes (a draft if the run fails to finish).
- `notes.trace.jsonl`: flushed structured events: exact model requests/responses,
  short decision summaries, tool arguments/results, final answer, and run status.
- `notes.stats.json`: finish status, steps/calls, latency, API spend.

Exit status `0` means the model chose to finish; `2` means the loop did not finish.
**Finishing is not a factual-correctness guarantee.** Existing outputs/traces are
never silently overwritten: use a new output path for each run.

## Your own audio/video

Install FFmpeg using your OS package manager (e.g. `sudo apt-get install ffmpeg`
on Ubuntu), then install the audio extra:

```bash
uv sync --frozen --extra audio --extra dev
uv run lecture-notes run \
  --media /path/to/lecture.mp4 --start 0 --end 180 \
  --task "Make concise study notes, preserving definitions and important examples." \
  --output runs/my-lecture/notes.md
```

This first runs local English ASR (`small.en`, CPU int8 by default), then the
model-directed notes loop. It also saves the timestamped transcript and ASR trace.
Video is accepted for its **audio track only**: slides/board text are not analyzed.
Use `--asr-model small` for multilingual Whisper; multilingual quality is not evaluated.
Prefer 2–10 minute sections for this MVP; whole-lecture coverage is not guaranteed.

For reusable ASR, independently of notes:

```bash
uv run lecture-notes transcribe /path/to/lecture.mp4 \
  --start 0 --end 180 --output runs/transcript.json
uv run lecture-notes run --transcript runs/transcript.json \
  --task "Make study notes." --output runs/notes.md
```

Optional terminology/context: `--context glossary.md` (UTF-8 text, maximum 12,000
characters). Context is explicitly marked as non-lecture evidence. PDF ingestion
is not implemented; the supplied conspect is used only as evaluation reference.

### GPU ASR

Add `--device cuda` when CUDA 12 and cuDNN 9 are installed. The tested machine
already has these libraries. On a clean CUDA-capable machine, one option is:

```bash
uv pip install nvidia-cublas-cu12==12.8.4.1 nvidia-cudnn-cu12==9.10.2.21
export LD_LIBRARY_PATH="$(uv run python -c 'import importlib.util; print(":".join(str(next(iter(importlib.util.find_spec(m).submodule_search_locations)))+"/lib" for m in ("nvidia.cublas", "nvidia.cudnn")))')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run lecture-notes transcribe /path/to/lecture.mp4 \
  --device cuda --output runs/gpu-transcript.json
```

If library discovery fails, use the CPU default; frozen-sample evaluation does not
need ASR or CUDA. ASR uses batched inference (GPU batch 16, CPU batch 4).

## Evaluation and tests

```bash
uv run pytest -q
uv run lecture-notes eval --output runs/eval-smoke --limit 2
uv run lecture-notes eval --output runs/eval-full --judge
```

The same frozen tasks and source intervals are given to the agent and one-call
baseline. Expected concepts are **not** passed to either. `--judge` adds a separate,
fallible local grounding/clarity assessment. It does not influence notes.
Evaluation emits per-task notes/traces/statistics, `results.json`, and `summary.json`.
Use a new output directory each time. See [the M2 report](docs/M2_REPORT.md) for
results, sample provenance, limitations, and three annotated failure traces.

**First measured results (20 section tasks):** agent finished 19/20; the
structure-and-coverage success proxy passed 19/20 versus baseline 15/20.
Lexical coverage was 97.5% versus 100%; mean generation latency 8.40 s versus
5.16 s. The loop helps formatting, but a factual-quality improvement is **not
established**. These are one-recording results, not general accuracy estimates.

If a long evaluation is interrupted, resume without losing completed tasks:

```bash
uv run lecture-notes eval --output runs/eval-full --judge --resume
```

Use the same sample/model/options. Incomplete attempts are archived under
`interrupted/`, not overwritten or counted as successful tasks.

## How this relates to M1

| M1 proposal | M2 implementation / reason |
|---|---|
| Qwen3-Omni-30B, Qwen3-ASR, VideoLLaMA | Qwen3 4B Q4_K_M via Ollama and Whisper small.en: faster setup and measurement within the time budget. Exact tested metadata is in `eval/results/runtime.json`; Ollama tags can change. |
| Act → reason → write → observe/revise | Model-selected `read_transcript`, `search_transcript`, `save_notes`, `finish`; draft feedback can trigger another read or revision. |
| Short-term memory and scratchpad | Visible conversation history, tool observations, current Markdown draft. No hidden chain-of-thought is requested or logged. |
| Long-term course knowledge base | Deferred; saved transcript/notes artifacts are not an automatic retrieval system. |
| Audio/video synchronization | Timestamped audio segments/citations only. Visual-only information remains a known limitation. |
| Speaker mapping, PDFs, web research | Deferred rather than superficially implemented. No diarization or web tools. |

Safety/recovery: tools have fixed capabilities, bounded observations, interval and
argument validation, recoverable tool errors/empty search results, a bounded model
retry, HTTP timeouts, and a maximum of 10 decisions by default (`--max-steps`).
The model cannot execute shell commands or choose arbitrary output paths.
Local API defaults to `http://127.0.0.1:11434`; override with `OLLAMA_HOST`.
Do not expose Ollama publicly; transcripts can contain private information.

## Recording and submission

Local models require a **3–5 minute narrated screen recording** of 2–3 frozen tasks
with traces being written. Follow [the recording guide](docs/RECORDING.md).
**Recording link: pending team recording/upload.** This is a remaining submission
requirement, not a completed deliverable.

Work is on `first-iteration`, as requested. **The rubric grades only commits on
`main` before the deadline**; the team must explicitly approve/integrate that step.

Dependencies, licenses, lecture/PDF sources: [THIRD_PARTY.md](THIRD_PARTY.md).
