import hashlib
import json
import math
from pathlib import Path


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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(segments):
    return "\n".join(f"[{timestamp(s['start'])}–{timestamp(s['end'])}] {s['text'].strip()}" for s in segments)


def transcribe(media, output, model="small.en", device="cpu", start=0, end=None):
    """faster-whisper (MIT): https://github.com/SYSTRAN/faster-whisper.

    Store the ASR output, not hidden reasoning. FFmpeg extracts a bounded clip.
    """
    import subprocess
    import tempfile
    from faster_whisper import WhisperModel

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
        segments, info = engine.transcribe(wav, beam_size=5, vad_filter=True, condition_on_previous_text=False)
        rows = []
        for segment in segments:
            rows.append({"start": round(start + segment.start, 3), "end": round(start + segment.end, 3),
                         "text": segment.text.strip(), "avg_logprob": segment.avg_logprob})
            print(f"[{timestamp(start + segment.start)}] {segment.text.strip()}", flush=True)
    if not rows:
        raise ValueError("ASR returned no speech; check the audio and selected interval.")
    data = {"source": str(media), "asr": {"model": model, "device": device, "language": info.language,
            "compute_type": "float16" if device == "cuda" else "int8"}, "segments": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data
