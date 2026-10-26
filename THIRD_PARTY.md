# Dependencies and sources

This project uses the documented APIs below; it does not vendor their source.

| Component | Purpose | License / source |
|---|---|---|
| Ollama | Local inference server | MIT, https://github.com/ollama/ollama |
| Qwen3-4B | Agent, baseline, optional judge | Apache-2.0, https://huggingface.co/Qwen/Qwen3-4B |
| faster-whisper | Local ASR | MIT, https://github.com/SYSTRAN/faster-whisper |
| Whisper small.en weights | English speech recognition | MIT, https://huggingface.co/Systran/faster-whisper-small.en |
| CTranslate2 | ASR inference | MIT, https://github.com/OpenNMT/CTranslate2 |
| PyAV | Audio decoding | BSD-3-Clause, https://github.com/PyAV-Org/PyAV |
| FFmpeg | Audio extraction | LGPL/GPL depending on installed build, https://ffmpeg.org/legal.html |
| yt-dlp | Optional lecture download | Unlicense (some bundled components differ), https://github.com/yt-dlp/yt-dlp |
| pytest | Tests | MIT, https://github.com/pytest-dev/pytest |
| pypdf | Development-only reference inspection | BSD-3-Clause, https://github.com/py-pdf/pypdf |

Architecture: a small model-directed act/observe loop with bounded execution and
an explicit operational decision summary. No external agent-framework code was copied.

## Evaluation material

- User-supplied lecture: **L1 Bayesian Methods in Machine Learning**, Materials of
  educational programs, https://www.youtube.com/watch?v=XwR767IfQHI .
- User-supplied conspect: **Байесовские методы в машинном обучении**, Д. П. Ветров,
  https://drive.google.com/file/d/1AQHXFYFA7V-Kt67lJGeEmzxdNkTk7KYJ/view .
  Lecture 1 is PDF pages 4–10 (printed pages 4–10).

No redistribution license was supplied for the lecture or PDF. We do not commit
the original audio/video or PDF. The evaluation contains limited ASR excerpts,
original task annotations, and derived outputs for this educational evaluation;
upstream rights remain with their owners. Do not interpret this notice as a
license to redistribute the original material. The PDF is reference-only, never
passed to the evaluated agent or baseline.
