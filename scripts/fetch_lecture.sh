#!/usr/bin/env bash
# yt-dlp (Unlicense): https://github.com/yt-dlp/yt-dlp
set -euo pipefail
mkdir -p downloads
uv run --extra audio yt-dlp --no-playlist -f bestaudio --no-progress \
  -o 'downloads/lecture.%(ext)s' 'https://www.youtube.com/watch?v=XwR767IfQHI'
