import datetime
import json
from pathlib import Path


class Trace:
    """Append-only, flushed JSONL. Requests include the exact visible context."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never silently append a new run to an existing trace.
        self.file = self.path.open("x", encoding="utf-8")
        self.sequence = 0

    def emit(self, event, **fields):
        self.sequence += 1
        row = {"seq": self.sequence, "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "event": event, **fields}
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
