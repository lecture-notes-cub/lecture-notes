import datetime
import json
from pathlib import Path


class Trace:
    """Append-only, flushed JSONL. Requests include the exact visible context."""

    def __init__(self, path, **defaults):
        self.path = Path(path)
        self.defaults = defaults
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never silently append a new run to an existing trace.
        self.file = self.path.open("x", encoding="utf-8")
        self.sequence = 0
        self.prompt_tokens = 0
        self.output_tokens = 0

    def emit(self, event, **fields):
        if event == "model_response" and isinstance(fields.get("response"), dict):
            self.prompt_tokens += fields["response"].get("prompt_eval_count", 0) or 0
            self.output_tokens += fields["response"].get("eval_count", 0) or 0
        self.sequence += 1
        row = {"seq": self.sequence, "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "event": event, **self.defaults, **fields}
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
