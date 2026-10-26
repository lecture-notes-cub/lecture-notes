import argparse
import json
import sys
from pathlib import Path

from .agent import run
from .model import Ollama
from .transcript import transcribe


def main():
    parser = argparse.ArgumentParser(description="Local lecture notes: transcribe, run agent, evaluate.")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("transcribe", help="Local ASR to reusable timestamped JSON")
    prep.add_argument("media", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--asr-model", default="small.en")
    prep.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    prep.add_argument("--start", type=float, default=0)
    prep.add_argument("--end", type=float)
    agent = sub.add_parser("run", help="Model-controlled notes agent (or one-call baseline)")
    source = agent.add_mutually_exclusive_group(required=True)
    source.add_argument("--transcript", type=Path)
    source.add_argument("--media", type=Path)
    agent.add_argument("--task", required=True)
    agent.add_argument("--output", type=Path, required=True)
    agent.add_argument("--start", type=float, default=0)
    agent.add_argument("--end", type=float)
    agent.add_argument("--context", type=Path, help="Optional UTF-8 glossary/context (.txt/.md)")
    agent.add_argument("--max-steps", type=int, default=10)
    agent.add_argument("--baseline", action="store_true")
    agent.add_argument("--asr-model", default="small.en")
    agent.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    agent.add_argument("--model", default="qwen3:4b")
    evaluate = sub.add_parser("eval", help="Frozen tasks: agent + baseline + metrics")
    evaluate.add_argument("--sample", type=Path, default=Path("eval/tasks.json"))
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--model", default="qwen3:4b")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--judge", action="store_true", help="Also run a local, fallible grounding/quality judge")
    args = parser.parse_args()
    try:
        if args.command == "transcribe":
            transcribe(args.media, args.output, args.asr_model, args.device, args.start, args.end)
        elif args.command == "run":
            if not 1 <= args.max_steps <= 30:
                parser.error("--max-steps must be between 1 and 30")
            transcript = args.transcript
            if args.media:
                transcript = args.output.with_suffix(".transcript.json")
                # ASR preparation is deterministic; the notes loop is model-controlled.
                transcribe(args.media, transcript, args.asr_model, args.device, args.start, args.end)
            context = args.context.read_text(encoding="utf-8") if args.context else ""
            if len(context) > 12000:
                parser.error("Context exceeds 12000 characters; supply a concise glossary.")
            stats = run(transcript, args.task, args.output, Ollama(args.model), args.start, args.end,
                        context, args.max_steps, args.baseline)
            print(json.dumps(stats, indent=2))
            if stats["status"] != "finished":
                sys.exit(2)
        else:
            from .evaluation import evaluate
            evaluate(args.sample, args.output, Ollama(args.model), args.limit, args.judge)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
