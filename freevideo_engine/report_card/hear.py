"""Words heard in a take, run as its own process in any Python that has faster-whisper (Report Card speech check).

The Report Card starts this file with FREEVIDEO_SPEECH_PYTHON, else the backtalk env on this PC, so ComfyUI's process
never loads ctranslate2. Reads {"audio": path of a float32 16 kHz mono .npy, "model": name} on stdin and writes {"words": [{text, start, end}]}.
Imports nothing from Eclipse; the model must already be on disk (local_files_only, nothing is downloaded).
"""
import json
import sys


def main():
    import numpy as np
    from faster_whisper import WhisperModel
    job = json.load(sys.stdin)
    audio = np.load(job['audio'])   # decoded by Eclipse: this env's own decoder may not match its PyAV
    model = WhisperModel(job.get('model') or 'small.en', device='cpu', compute_type='int8', local_files_only=True)
    segments, _ = model.transcribe(audio, word_timestamps=True, vad_filter=True, beam_size=5)
    words = [dict(text=w.word.strip(), start=round(w.start, 3), end=round(w.end, 3))
             for segment in segments for w in (segment.words or ()) if w.word.strip()]
    json.dump(dict(words=words), sys.stdout)


if __name__ == '__main__':
    main()
