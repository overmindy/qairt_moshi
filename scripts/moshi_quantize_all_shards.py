"""Submit W8A16 quantization and S26 compilation for existing Moshi uploads."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--compiled-shards", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    entries = json.loads(args.compiled_shards.read_text())
    manifest = json.loads((args.export_dir / "manifest.json").read_text())
    by_range = {(item["start_layer"], item["end_layer_exclusive"]): item for item in manifest["shards"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries[args.start:]:
        start, end = entry["start"], entry["end"]
        shard = by_range[(start, end)]["onnx"]
        output = args.output_dir / f"layers_{start}_{end}_w8a16.json"
        if output.exists():
            print(f"skip existing {output}", flush=True)
            continue
        command = [sys.executable, str(Path(__file__).with_name("moshi_quantize_shard.py")),
                   "--export-dir", str(args.export_dir), "--shard", shard,
                   "--source-model-id", entry["model_id"], "--samples", str(args.samples),
                   "--output", str(output)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
