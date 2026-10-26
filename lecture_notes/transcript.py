import hashlib
import json
import math
import time
from pathlib import Path

from .trace import Trace


def timestamp(seconds):
    value = int(seconds)
    return f"{value // 3600:02d}:{value % 3600 // 60:02d}:{value % 60:02d}"


def load_transcript(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = data.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Transcript must contain a nonempty 'segments' list.")
    previous = -1
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if (not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start or start < previous
                or not isinstance(segment["text"], str)):
            raise ValueError("Invalid transcript segment: need ordered finite times and text.")
        previous = start
    return data


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(segments):
    return "\n".join(f"[{timestamp(s['start'])}–{timestamp(s['end'])}] {s['text'].strip()}" for s in segments)


def transcribe(media, output, model="small.en", device="cpu", start=0, end=None):
    if not math.isfinite(start) or start < 0 or (end is not None and (not math.isfinite(end) or end <= start)):
        raise ValueError("Require finite 0 <= start < end.")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite transcript: {output}")
    trace = Trace(output.with_suffix(".asr.trace.jsonl"))
    began = time.monotonic()
    try:
        trace.emit("asr_request", media=str(media), media_sha256=fingerprint(media),
                   model=model, device=device, start=start, end=end,
                   beam_size=5, batch_size=16 if device == "cuda" else 4,
                   vad_filter=True, condition_on_previous_text=False)
        data = _transcribe(media, output, model, device, start, end)
        trace.emit("asr_result", latency_s=time.monotonic() - began, transcript=data,
                   output_sha256=fingerprint(output))
        return data
    except Exception as exc:
        trace.emit("asr_error", error=str(exc), latency_s=time.monotonic() - began)
        raise
    finally:
        trace.close()


def _transcribe(media, output, model, device, start, end):
    """faster-whisper (MIT): https://github.com/SYSTRAN/faster-whisper.

    Store the ASR output, not hidden reasoning. FFmpeg extracts a bounded clip.
    """
    import subprocess
    import tempfile
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite transcript: {output}")
    if start < 0 or (end is not None and end <= start):
        raise ValueError("Require 0 <= start < end.")
    with tempfile.TemporaryDirectory(prefix="lecture-notes-") as directory:
        wav = str(Path(directory) / "audio.wav")
        command = ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", str(media)]
        if end is not None:
            command += ["-t", str(end - start)]
        command += ["-vn", "-ac", "1", "-ar", "16000", wav]
        subprocess.run(command, check=True, timeout=600)
        engine = WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8",
                              cpu_threads=8)
        pipeline = BatchedInferencePipeline(model=engine)
        segments, info = pipeline.transcribe(wav, beam_size=5, vad_filter=True,
                                            condition_on_previous_text=False,
                                            batch_size=16 if device == "cuda" else 4)
        rows = []
        for segment in segments:
            rows.append({"start": round(start + segment.start, 3), "end": round(start + segment.end, 3),
                         "text": segment.text.strip(), "avg_logprob": segment.avg_logprob})
            print(f"[{timestamp(start + segment.start)}] {segment.text.strip()}", flush=True)
    if not rows:
        raise ValueError("ASR returned no speech; check the audio and selected interval.")
    data = {"source": str(media), "asr": {"model": model, "device": device, "language": info.language,
            "compute_type": "float16" if device == "cuda" else "int8",
            "batch_size": 16 if device == "cuda" else 4,
            "beam_size": 5, "vad_filter": True, "condition_on_previous_text": False}, "segments": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data
