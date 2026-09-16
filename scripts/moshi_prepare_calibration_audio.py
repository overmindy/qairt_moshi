"""Build a tiny deterministic 24 kHz Moshi smoke/calibration audio set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import wave
from array import array
from collections import defaultdict
from pathlib import Path

SAMPLE_RATE = 24_000
FRAME_SAMPLES = 1_920
HF_DUMMY_NAME = "hf-internal-testing/librispeech_asr_dummy"
HF_DUMMY_CONFIG = "clean"
HF_DUMMY_URL = f"https://huggingface.co/datasets/{HF_DUMMY_NAME}"
LIBRISPEECH_URL = "https://www.openslr.org/12/"
DATASET_LICENSE = "CC BY 4.0 (LibriSpeech source corpus)"


def _read_wav(path: Path) -> array:
    with wave.open(str(path), "rb") as source:
        properties = (
            source.getnchannels(),
            source.getsampwidth(),
            source.getframerate(),
            source.getcomptype(),
        )
        expected = (1, 2, SAMPLE_RATE, "NONE")
        if properties != expected:
            raise ValueError(f"{path}: expected {expected}, got {properties}")
        samples = array("h")
        samples.frombytes(source.readframes(source.getnframes()))
        if sys.byteorder != "little":
            samples.byteswap()
        return samples


def _write_wav(path: Path, samples: array) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(samples.tobytes())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_clip(samples: array, length: int, rng: random.Random) -> array:
    if len(samples) > length:
        start = rng.randrange(len(samples) - length + 1)
        return samples[start : start + length]
    result = array("h", samples)
    result.extend([0] * (length - len(result)))
    return result


def _mix_with_delay(first: array, second: array, delay: int) -> array:
    result = array("h", [0]) * len(first)
    for index, left in enumerate(first):
        right_index = index - delay
        right = second[right_index] if 0 <= right_index < len(second) else 0
        mixed = round(0.62 * left + 0.38 * right)
        result[index] = max(-32768, min(32767, mixed))
    return result


def _transcript(path: Path) -> str:
    for suffix in (".normalized.txt", ".original.txt"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate.read_text().strip()
    return ""


def _entry(
    output_dir: Path,
    path: Path,
    variant: str,
    sources: list[str],
    speakers: list[str],
    transcripts: list[str],
    frames: int,
) -> dict[str, object]:
    return {
        "id": path.stem,
        "path": str(path.relative_to(output_dir)),
        "variant": variant,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "pcm_bits": 16,
        "moshi_frames": frames,
        "duration_seconds": frames * FRAME_SAMPLES / SAMPLE_RATE,
        "speaker_ids": speakers,
        "transcripts": transcripts,
        "source_files": sources,
        "sha256": _sha256(path),
    }


def _write_jsonl(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in entries)
    )


def _decoder_to_pcm16(decoder: object) -> array:
    """Decode a datasets 4.x AudioDecoder after its 24 kHz cast."""
    samples = decoder.get_all_samples()  # type: ignore[attr-defined]
    if int(samples.sample_rate) != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {samples.sample_rate}")
    data = samples.data.detach().cpu().float()
    if data.ndim == 2:
        data = data.mean(dim=0)
    if data.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {tuple(data.shape)}")
    # Avoid relying on NumPy: PyTorch and datasets are already dependencies.
    return array(
        "h",
        (
            max(-32768, min(32767, round(value * 32767)))
            for value in data.clamp(-1, 1).tolist()
        ),
    )


def _load_hf_dummy(
    count: int, seed: int
) -> tuple[list[dict[str, object]], dict[str, str]]:
    try:
        from datasets import Audio, load_dataset
    except ImportError as error:
        raise SystemExit(
            "The tiny Hugging Face dataset path needs `datasets` and its audio "
            "dependencies. Install the repository requirements first."
        ) from error

    dataset = load_dataset(
        HF_DUMMY_NAME,
        HF_DUMMY_CONFIG,
        split="validation",
    ).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    if count > len(dataset):
        raise SystemExit(
            f"Requested {count} clean clips, but dummy set has {len(dataset)}"
        )
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    rows: list[dict[str, object]] = []
    for index in indices[:count]:
        row = dataset[index]
        rows.append(
            {
                "samples": _decoder_to_pcm16(row["audio"]),
                "speaker": str(row["speaker_id"]),
                "transcript": str(row["text"]),
                "source": str(row["id"]),
            }
        )
    return rows, {
        "dataset": HF_DUMMY_NAME,
        "dataset_url": HF_DUMMY_URL,
        "source_corpus_url": LIBRISPEECH_URL,
        "license": DATASET_LICENSE,
    }


def _load_libritts(
    source_dir: Path, count: int, seed: int
) -> tuple[list[dict[str, object]], dict[str, str]]:
    if not source_dir.is_dir():
        raise SystemExit(f"Missing extracted LibriTTS directory: {source_dir}")
    by_speaker: dict[str, list[Path]] = defaultdict(list)
    for source in sorted(source_dir.rglob("*.wav")):
        relative = source.relative_to(source_dir)
        if len(relative.parts) >= 3:
            by_speaker[relative.parts[0]].append(source)
    speakers = sorted(by_speaker)
    if count > len(speakers):
        raise SystemExit(f"Requested {count} speakers, but found only {len(speakers)}")
    rng = random.Random(seed)
    selected_speakers = rng.sample(speakers, count)
    rows = []
    for speaker in selected_speakers:
        source = rng.choice(by_speaker[speaker])
        rows.append(
            {
                "samples": _read_wav(source),
                "speaker": speaker,
                "transcript": _transcript(source),
                "source": str(source.relative_to(source_dir)),
            }
        )
    return rows, {
        "dataset": "LibriTTS train-clean-100",
        "dataset_url": "https://www.openslr.org/60/",
        "license": "CC BY 4.0",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("hf-dummy", "libritts"), default="hf-dummy"
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clean-count", type=int, default=8)
    parser.add_argument("--overlap-count", type=int, default=1)
    parser.add_argument("--silence-count", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("Use an empty --output-dir")
    if args.clean_count < 2 or args.overlap_count < 0 or args.silence_count < 0:
        raise SystemExit("Invalid sample counts")
    frames = round(args.seconds * SAMPLE_RATE / FRAME_SAMPLES)
    if frames < 2:
        raise SystemExit("Clip duration must contain at least two Moshi frames")
    length = frames * FRAME_SAMPLES

    rng = random.Random(args.seed)
    if args.source == "hf-dummy":
        rows, provenance = _load_hf_dummy(args.clean_count, args.seed)
    else:
        if args.source_dir is None:
            raise SystemExit("--source-dir is required with --source libritts")
        rows, provenance = _load_libritts(
            args.source_dir, args.clean_count, args.seed
        )
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True)
    entries: list[dict[str, object]] = []
    clean_samples: list[array] = []
    clean_transcripts: list[str] = []

    selected_sources: list[str] = []
    selected_speakers: list[str] = []
    for index, row in enumerate(rows):
        samples = _fixed_clip(row["samples"], length, rng)  # type: ignore[arg-type]
        transcript = str(row["transcript"])
        speaker = str(row["speaker"])
        source = str(row["source"])
        destination = audio_dir / f"clean_{index:03d}_{source}.wav"
        _write_wav(destination, samples)
        entries.append(
            _entry(
                args.output_dir,
                destination,
                "clean",
                [source],
                [speaker],
                [transcript],
                frames,
            )
        )
        clean_samples.append(samples)
        clean_transcripts.append(transcript)
        selected_sources.append(source)
        selected_speakers.append(speaker)

    delay = round(0.48 * SAMPLE_RATE)
    for index in range(args.overlap_count):
        left = index % len(clean_samples)
        right = (index * 7 + 1) % len(clean_samples)
        if right == left:
            right = (right + 1) % len(clean_samples)
        samples = _mix_with_delay(clean_samples[left], clean_samples[right], delay)
        destination = audio_dir / f"overlap_{index:03d}.wav"
        _write_wav(destination, samples)
        entries.append(
            _entry(
                args.output_dir,
                destination,
                "overlap",
                [selected_sources[left], selected_sources[right]],
                [selected_speakers[left], selected_speakers[right]],
                [clean_transcripts[left], clean_transcripts[right]],
                frames,
            )
        )

    for index in range(args.silence_count):
        destination = audio_dir / f"silence_{index:03d}.wav"
        _write_wav(destination, array("h", [0]) * length)
        entries.append(
            _entry(
                args.output_dir,
                destination,
                "silence",
                [],
                [],
                [],
                frames,
            )
        )

    clean_entries = [item for item in entries if item["variant"] == "clean"]
    overlap_entries = [item for item in entries if item["variant"] == "overlap"]
    silence_entries = [item for item in entries if item["variant"] == "silence"]
    smoke = clean_entries[:2] + overlap_entries[:1] + silence_entries[:1]
    _write_jsonl(args.output_dir / "calibration.jsonl", entries)
    _write_jsonl(args.output_dir / "smoke.jsonl", smoke)
    summary = {
        **provenance,
        "seed": args.seed,
        "source_dir": str(args.source_dir.resolve()) if args.source_dir else None,
        "counts": {
            "clean": len(clean_entries),
            "overlap": len(overlap_entries),
            "silence": len(silence_entries),
            "total": len(entries),
            "smoke": len(smoke),
        },
        "moshi_frames_per_clip": frames,
        "duration_seconds_per_clip": frames * FRAME_SAMPLES / SAMPLE_RATE,
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
