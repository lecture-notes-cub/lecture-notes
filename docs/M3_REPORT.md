# M3 report

## Setup

We compared three agent variants and a one-call baseline on the unchanged 20-task
M2 sample. All runs
used Qwen3 4B Q4_K_M through Ollama 0.40.16, temperature 0, seed 42 and a
16,384-token context on one NVIDIA A100 80 GB.

Following the M2 feedback, we manually checked all ten key-claim
descriptions, numerical values and evidence intervals for t01, t03, t17 and t19
against the frozen transcript before evaluation. 

## Variants and hypotheses

| Variant | Mechanism | Hypothesis and source |
|---|---|---|
| `m2_loop` | M2 ReAct loop: model chooses reads, searches, saves and finish. | Flexible but likely to waste calls on empty searches. M2 and L2–L5 ReAct/tool-use patterns. |
| `lean_loop` | Code hides search; one full read, at most two saves and four decisions. | Preserve quality with fewer calls/tokens; long/noisy t08 and t16 may suffer. L2–L5 plan-and-execute, tool hiding and stop rules. |
| `claim_check` | Code requires `check_claims` after every save before finish. | Improve grounded claims at the cost of an extra call. L2–L3 self-check and L5 finish gate. |
| `one_call_baseline` | One call receives the full interval and writes notes. | Non-agent cost/quality reference. |

## Results

M3 success requires finish, structural validity, and either all key-claim checks on
the four labeled tasks or at least 75% M2 lexical coverage on the other tasks. This
is a pre-registered proxy, not semantic ground truth. Tokens are prompt plus output
tokens from Ollama.


| Variant | Finish | Structure | M3 success | Key claims | Lexical | Steps / model calls | Tool calls | Tokens/task (in + out) | Latency/task | Judge faithfulness | Clarity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `m2_loop` | 20/20 | 20/20 | **18/20 (90%)** | 66.7% | 96.3% | 4.15 | 4.15 | 7,092 (5,956 + 1,136) | 10.65 s | **94.2%** | 3.80 |
| `lean_loop` | 16/20 | 16/20 | 15/20 (75%) | **91.7%** | 77.5% | 3.15 | 3.00 | 5,455 (4,081 + 1,374) | 9.28 s | 91.7% | 3.75 |
| `claim_check` | 5/20 | 20/20 drafts | 4/20 (20%) | 66.7% | 96.3% | 9.65 | 9.65 | 38,942 (35,922 + 3,020) | 21.43 s | 93.5% | **3.85** |
| `one_call_baseline` | 20/20 | 17/20 | 16/20 (80%) | 79.2% | **98.8%** | **1.00** | **0** | **1,681 (902 + 779)** | **5.48 s** | 87.9% | 3.65 |

The first `m2_loop` task includes a 59.04-second cold model load; excluding only that
task gives 8.10 seconds over its other 19 tasks. Thus the raw 10.65-second mean is
not a fair pure-generation comparison. Calls and tokens are unaffected. Judge means
cover 20 drafts except `lean_loop`, where only its 16 produced drafts were judged.

### Per-task breakdown

Each cell is `M3 success; model calls/steps; tokens; generation latency`. One model
call occurs per step in this harness. Dollar cost is $0 for every cell. Full metrics,
notes, stats and traces are in `eval/m3_results/`.

| Task | `m2_loop` | `lean_loop` | `claim_check` | Baseline |
|---|---|---|---|---|
| t01 | ✓; 4c; 4,624t; 59.04s | ✓; 3c; 3,368t; 4.01s | ✓; 9c; 20,532t; 10.50s | ✓; 1c; 1,154t; 3.83s |
| t02 | ✓; 3c; 2,783t; 3.97s | ✓; 3c; 3,440t; 5.45s | ✓; 8c; 15,089t; 9.18s | ✓; 1c; 963t; 3.88s |
| t03 | ✗; 3c; 3,692t; 5.55s | ✓; 3c; 4,175t; 6.37s | ✗; 10c; 35,753t; 19.26s | ✗; 1c; 1,155t; 3.85s |
| t04 | ✓; 5c; 7,000t; 7.29s | ✓; 4c; 6,520t; 8.40s | ✗; 10c; 38,933t; 25.16s | ✓; 1c; 1,157t; 3.48s |
| t05 | ✓; 3c; 4,173t; 6.87s | ✗; 2c; 4,905t; 20.72s | ✗; 10c; 28,350t; 14.69s | ✓; 1c; 1,146t; 3.78s |
| t06 | ✓; 5c; 7,484t; 7.58s | ✗; 4c; 4,390t; 3.80s | ✗; 10c; 34,860t; 19.49s | ✓; 1c; 999t; 3.11s |
| t07 | ✓; 5c; 6,799t; 6.11s | ✓; 3c; 3,573t; 4.55s | ✗; 10c; 27,558t; 23.87s | ✓; 1c; 1,298t; 4.76s |
| t08 | ✓; 4c; 6,029t; 7.36s | ✓; 3c; 5,364t; 7.09s | ✗; 10c; 45,773t; 29.10s | ✓; 1c; 2,003t; 12.47s |
| t09 | ✓; 4c; 6,045t; 6.28s | ✗; 4c; 5,167t; 20.00s | ✗; 10c; 40,097t; 20.16s | ✓; 1c; 1,364t; 3.88s |
| t10 | ✓; 5c; 8,076t; 10.93s | ✓; 3c; 3,732t; 5.14s | ✗; 10c; 30,566t; 19.10s | ✓; 1c; 987t; 3.09s |
| t11 | ✓; 3c; 4,683t; 6.02s | ✓; 3c; 4,915t; 6.23s | ✗; 10c; 43,846t; 23.89s | ✗; 1c; 1,597t; 4.10s |
| t12 | ✓; 4c; 6,252t; 6.45s | ✓; 4c; 9,194t; 11.80s | ✓; 9c; 31,792t; 15.91s | ✓; 1c; 2,002t; 6.62s |
| t13 | ✓; 4c; 7,553t; 7.31s | ✓; 3c; 5,368t; 6.17s | ✗; 10c; 44,286t; 20.53s | ✓; 1c; 1,906t; 4.68s |
| t14 | ✓; 3c; 4,015t; 5.45s | ✓; 3c; 4,683t; 6.67s | ✓; 9c; 34,217t; 23.87s | ✓; 1c; 1,343t; 4.30s |
| t15 | ✓; 4c; 9,146t; 11.65s | ✓; 4c; 10,231t; 13.74s | ✗; 10c; 43,683t; 22.57s | ✓; 1c; 2,217t; 6.84s |
| t16 | ✓; 4c; 11,057t; 14.98s | ✗; 2c; 4,416t; 20.54s | ✗; 10c; 60,314t; 27.56s | ✓; 1c; 2,845t; 9.17s |
| t17 | ✓; 4c; 9,374t; 9.58s | ✓; 3c; 7,157t; 9.70s | ✗; 10c; 49,813t; 21.59s | ✓; 1c; 3,108t; 10.26s |
| t18 | ✓; 7c; 13,387t; 9.83s | ✓; 3c; 5,656t; 8.07s | ✗; 10c; 38,838t; 18.03s | ✓; 1c; 1,984t; 6.08s |
| t19 | ✗; 5c; 11,364t; 14.17s | ✗; 3c; 6,021t; 9.13s | ✗; 8c; 58,903t; 38.40s | ✗; 1c; 1,985t; 6.24s |
| t20 | ✓; 4c; 8,307t; 6.55s | ✓; 3c; 6,825t; 7.97s | ✗; 10c; 55,633t; 25.74s | ✗; 1c; 2,399t; 5.27s |

### Hypothesis outcomes

- `lean_loop` used 24% fewer calls and 23% fewer tokens than M2, but lost four
  completions. The predicted long-task risk appeared on t16; other failures came
  from a forbidden reread, finishing without a draft and exhausting output.
- `claim_check` rejected its hypothesis: it used 5.5× the M2 tokens without
  improving key-claim score. Revisions repeatedly invalidated checks until the cap.
- `m2_loop` retained wasteful searches but recovered from invalid decisions,
  finished every task and achieved the highest success and judge faithfulness.

## Repeatability (pass^3)

Every configuration was repeated twice without the judge because the judge does not
affect the pre-registered M3 success result. We define **pass^3** as passing all
three runs and pass@3 as passing at least one of the three.

| Variant | Run 1 | Run 2 | Run 3 | pass^3 | pass@3 | Never passed | Unstable tasks |
|---|---:|---:|---:|---:|---:|---|---|
| `m2_loop` | 18/20 | 18/20 | 18/20 | **17/20 (85%)** | **19/20 (95%)** | t19 | t03, t17 |
| `lean_loop` | 15/20 | 14/20 | 15/20 | 14/20 (70%) | 15/20 (75%) | t05, t06, t09, t16, t19 | t01 |
| `claim_check` | 4/20 | 9/20 | 10/20 | 1/20 (5%) | 13/20 (65%) | t03, t04, t09, t15, t16, t17, t19 | 12 other tasks |
| `one_call_baseline` | 16/20 | 16/20 | 16/20 | 16/20 (80%) | 16/20 (80%) | t03, t11, t19, t20 | none |

The baseline is perfectly repeatable under this metric. Lean is also mostly stable,
but five mechanism-linked failures never recover. Claim-check has the widest gap
between pass^3 and pass@3: it can pass 13 tasks at least once, but only one task
passes reliably in all runs. Its finish-gate interaction is therefore not merely
expensive; it is highly unstable.

For the selected winner, per-run cost was:

| Run | Success | Failed tasks | Calls/task | Tokens/task | Latency/task |
|---|---:|---|---:|---:|---:|
| Main | 18/20 | t03, t19 | 4.15 | 7,092 | 10.65 s (cold start included) |
| Repeat 2 | 18/20 | t17, t19 | 3.65 | 5,923 | 7.19 s |
| Repeat 3 | 18/20 | t17, t19 | 3.70 | 6,012 | 7.31 s |

For `m2_loop`, t19 failed all runs; t03 and t17 changed outcome despite the fixed
seed. Across variants, the mechanism-level differences remain larger than its
within-agent variation, especially for claim-check completion and cost. These results
confirm the M2 feedback that one run is not enough: a single claim-check run reported
20% success, while its other runs reached 45% and 50%, yet strict pass^3 was only 5%.

## Rational selection

The winner is **`m2_loop`**. Our performance measure prioritizes usable, grounded
notes and treats compute as a secondary cost. `m2_loop` has the highest M3 success
(90%), full completion/structure, the highest judge faithfulness (94.2%), and the
highest pass^3 (85%). `lean_loop` saves about one quarter of calls/tokens but loses three M3
successes and four completions. `claim_check` is dominated on both success and cost.
The baseline is the cheapest and remains rational when throughput is the primary
objective, but for the stated student-facing task its 80% success, 17/20 structural
pass and lower judge faithfulness do not justify selecting it over `m2_loop`.

This is a rational choice relative to the stated measure, not a claim that the loop
is universally best. If future human semantic labels show that the extra loop calls
do not improve factual usefulness, the one-call baseline should replace it.

## Reproduction

```bash
uv run lecture-notes eval --output eval/m3_results --judge --timeout 600
uv run lecture-notes eval --output eval/m3_passk_run2 --variants m2_loop --timeout 600
uv run lecture-notes eval --output eval/m3_passk_run3 --variants m2_loop --timeout 600
uv run lecture-notes eval --output eval/m3_others_passk_run2 \
  --variants lean_loop claim_check one_call_baseline --timeout 600
uv run lecture-notes eval --output eval/m3_others_passk_run3 \
  --variants lean_loop claim_check one_call_baseline --timeout 600
```

Artifacts are under `eval/m3_*`; every JSONL record contains its variant name.
See `docs/M3_RECORDING_SCRIPT.md` for the local-model video script.
