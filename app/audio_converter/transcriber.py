import argparse
import json
import os
import sys
from typing import Any, Dict, List
from urllib.parse import urlparse, parse_qs


def ensure_dependencies() -> None:
    """Check that required libraries are installed and print install help if not.

    Required packages:
      - openai-whisper (backbone model)
      - stable-ts (word-level timestamps)
      - torch (required by whisper)
      - soundfile (I/O for certain formats)
    """
    missing: List[str] = []
    try:
        import whisper  # noqa: F401
    except Exception:
        missing.append("openai-whisper")
    try:
        import stable_whisper  # noqa: F401
    except Exception:
        missing.append("stable-ts")
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch")
    try:
        import soundfile  # noqa: F401
    except Exception:
        missing.append("soundfile")

    if missing:
        print(
            "Missing dependencies: " + ", ".join(missing) +
            "\nInstall with:\n\n  pip install openai-whisper stable-ts torch soundfile\n",
            file=sys.stderr,
        )


def load_model(model_name: str = "small"):
    """Load a Whisper model (via stable-ts for word-level timestamps)."""
    import stable_whisper

    try:
        model = stable_whisper.load_model(model_name)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_name}': {exc}")


def extract_youtube_id(video_url: str) -> str | None:
    """Extract a YouTube video ID from common URL formats."""
    try:
        parsed = urlparse(video_url)
        host = (parsed.netloc or "").lower()
        path = parsed.path or ""
        if host.endswith("youtu.be"):
            vid = path.strip("/")
            return vid or None
        if path == "/watch":
            qs = parse_qs(parsed.query or "")
            vids = qs.get("v")
            if vids and vids[0]:
                return vids[0]
        if path.startswith("/shorts/"):
            return path.split("/shorts/")[-1].split("/")[0]
        last = path.strip("/").split("/")[-1]
        return last or None
    except Exception:
        return None

def transcribe_to_words(audio_path: str, model_name: str = "small", language_hint: str = None) -> List[Dict[str, Any]]:
    """Transcribe audio and return a list of word dicts per spec.

    Each dict has: {"word": str, "start_time": float, "end_time": float}

    - language_hint: e.g., 'te' to bias Telugu; None will auto-detect and preserve original scripts
      (Telugu and Sanskrit slokas are kept in their native scripts).
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = load_model(model_name)

    try:
        result = model.transcribe(
            audio_path,
            task="transcribe",
            language=language_hint,  # None => auto-detect
            word_timestamps=True,
            vad=True,
            condition_on_previous_text=True,
            temperature=[0.0, 0.2, 0.4],
            verbose=False,
        )
        rd = result.to_dict()
    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}")

    words: List[Dict[str, Any]] = []
    for seg in rd.get("segments", []) or []:
        for w in seg.get("words", []) or []:
            text = (w.get("word") or "").strip()
            if not text:
                continue
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                continue
            words.append({
                "word": text,
                "start_time": float(start),
                "end_time": float(end),
            })
    return words


def write_output_json(words: List[Dict[str, Any]], video_url: str, thumbnail_url: str, output_path: str) -> None:
    data = {
        "video_url": video_url,
        "thumbnail_url": thumbnail_url,
        "transcript": words,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    ensure_dependencies()

    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an audio file (Telugu + Sanskrit supported) with openai-whisper "
            "and output a combined JSON containing video and thumbnail URLs."
        )
    )
    parser.add_argument("audio_path", type=str, help="Path to the audio/video file")
    parser.add_argument("video_url", type=str, help="Full YouTube video URL")
    parser.add_argument("thumbnail_url", type=str, nargs='?', default=None, help="Optional thumbnail URL; derived from YouTube URL if omitted")
    parser.add_argument("--model", type=str, default="small", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size (default: small)")
    parser.add_argument("--language", type=str, default=None, help="Optional language hint (e.g., 'te' for Telugu). Default: auto-detect")
    parser.add_argument("--out", type=str, default="output.json", help="Output JSON filename (default: output.json)")

    args = parser.parse_args()

    # Basic validation
    if not args.audio_path or not args.video_url:
        print("Usage: python transcriber.py <audio_path> <video_url> [thumbnail_url]", file=sys.stderr)
        sys.exit(2)

    try:
        words = transcribe_to_words(args.audio_path, model_name=args.model, language_hint=args.language)
        output_path = os.path.abspath(args.out)
        # Derive thumbnail if not provided
        thumbnail_url = args.thumbnail_url
        if not thumbnail_url:
            vid = extract_youtube_id(args.video_url)
            if not vid:
                raise RuntimeError("Unable to derive thumbnail: could not extract video ID from the YouTube URL")
            thumbnail_url = f"https://img.youtube.com/vi/{vid}/hqdefault.jpg"
        write_output_json(words, args.video_url, thumbnail_url, output_path)
        print(f"Saved: {output_path}")
    except FileNotFoundError as fnf:
        print(str(fnf), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as rte:
        print(str(rte), file=sys.stderr)
        sys.exit(3)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def ensure_dependencies() -> None:
    """Best-effort check that required libraries are installed.

    We rely on:
      - openai-whisper (model loading and transcription backbone)
      - stable-ts (word-level timestamps on top of Whisper)
      - torch (required by Whisper)
      - soundfile (I/O for certain audio formats)
    """
    missing: List[str] = []
    try:
        import whisper  # noqa: F401
    except Exception:
        missing.append("openai-whisper")
    try:
        import stable_whisper  # noqa: F401
    except Exception:
        missing.append("stable-ts")
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch")
    try:
        import soundfile  # noqa: F401
    except Exception:
        missing.append("soundfile")

    if missing:
        print(
            "Missing dependencies detected: " + ", ".join(missing) +
            "\nInstall them with:\n\n"
            "  pip install openai-whisper stable-ts torch soundfile\n",
            file=sys.stderr,
        )
        # Do not exit immediately; allow user to read the message if they invoked help.


def load_model(model_name: str = "small"):
    """Load a Whisper model using stable-ts wrapper for word-level timestamps.

    On first use, models are downloaded. Choose from: tiny, base, small, medium, large.
    """
    import stable_whisper

    try:
        # stable-ts provides a drop-in that returns extended structures with word timings
        model = stable_whisper.load_model(model_name)
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_name}': {exc}")


def transcribe_with_words(model, audio_path: str, language_hint: str = None) -> Dict[str, Any]:
    """Transcribe the audio and return detailed result including words with timestamps.

    - language_hint: optional BCP-47 code (e.g., 'te' for Telugu). If None, Whisper auto-detects.
    - We keep task='transcribe' to preserve original scripts for Telugu and Sanskrit.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        result = model.transcribe(
            audio_path,
            task="transcribe",
            language=language_hint,  # None -> auto-detect; use 'te' to bias Telugu
            word_timestamps=True,
            vad=True,  # helps with more accurate word segmentation
            condition_on_previous_text=True,
            # Temperature settings can help with mixed-language accuracy
            temperature=[0.0, 0.2, 0.4],
            verbose=False,
        )
        # Convert to dict to easily iterate words
        return result.to_dict()
    except Exception as exc:
        raise RuntimeError(f"Transcription failed: {exc}")


def extract_word_level(result_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten word-level data from the stable-ts result dict.

    Each item: {"text": str, "start": float, "end": float}
    """
    words: List[Dict[str, Any]] = []
    for seg in result_dict.get("segments", [])
:
        for w in seg.get("words", []) or []:
            # stable-ts word object contains 'word', 'start', 'end'
            text = (w.get("word") or "").strip()
            start = w.get("start")
            end = w.get("end")
            if text:
                words.append({"text": text, "start": float(start), "end": float(end)})
    return words


def pretty_print(words: List[Dict[str, Any]]) -> None:
    """Print words in a readable JSON format."""
    print(json.dumps(words, ensure_ascii=False, indent=2))


def main() -> None:
    ensure_dependencies()

    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an audio/video file using OpenAI Whisper (via stable-ts) "
            "with word-level timestamps. Handles Telugu and Sanskrit in original scripts."
        )
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the input audio/video file (e.g., .wav, .mp3, .mp4).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to use (default: small)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help=(
            "Optional language hint (e.g., 'te' for Telugu). "
            "Leave empty to auto-detect (recommended for mixed Telugu/Sanskrit)."
        ),
    )

    args = parser.parse_args()

    try:
        model = load_model(args.model)
        result_dict = transcribe_with_words(model, args.audio_path, args.language)
        words = extract_word_level(result_dict)

        # Print JSON list of {text, start, end}
        pretty_print(words)
    except FileNotFoundError as fnf:
        print(str(fnf), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as rte:
        print(str(rte), file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()


