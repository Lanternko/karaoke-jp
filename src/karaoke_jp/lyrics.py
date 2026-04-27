"""M3: vocals -> ASR -> mora-aligned ruby LRC. Stub."""
from __future__ import annotations

# TODO: implement at M3. Pipeline:
#   1. mlx-whisper (Mac) or faster-whisper (Linux GPU) -> raw text
#   2. WhisperX wav2vec2 alignment -> word timestamps
#   3. SOFA + opencpop-cjke-multidict -> mora timestamps
#   4. fugashi + UniDic -> tokens
#   5. ruby.add_furigana (yomikata + override JSON)
#   6. emit ruby.lrc
