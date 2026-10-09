"""Download only missing guarded rebuild sources, on a separate remote volume."""
import argparse
import json
import shutil
from pathlib import Path

from moshi_rebuild_guarded_dlc_set import (
    _sha256, _write_json, download_resumable, extract_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--graph", action="append", required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline_manifest.read_text())
    args.cache_root.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in args.graph:
        try:
            model_id = baseline["graphs"][name]["quantized_model_id"]
            receipt_path = args.output_root / "graphs" / name / "source.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                source = Path(receipt["onnx"])
                if (receipt["model_id"] != model_id or not source.is_file()
                        or _sha256(source) != receipt["onnx_sha256"]):
                    raise ValueError("Existing source receipt failed verification")
                print(f"{name}: REUSE", flush=True)
                continue
            root = args.cache_root / name
            root.mkdir(exist_ok=True)
            if shutil.disk_usage(root).free < 12 * 1024**3:
                raise RuntimeError("Need at least 12 GiB free on cache volume")
            archive = root / f"{model_id}.onnx.zip"
            print(f"{name}: downloading {model_id}", flush=True)
            if not archive.exists():
                download_resumable(model_id, archive)
            source = extract_model(archive, root / "source")
            _write_json(receipt_path, {
                "model_id": model_id, "onnx": str(source),
                "onnx_sha256": _sha256(source), "archive_sha256": _sha256(archive),
            })
            print(f"{name}: DOWNLOAD VERIFIED {source}", flush=True)
        except Exception as error:
            failures.append(name)
            print(f"{name}: FAILED {type(error).__name__}: {error}", flush=True)
    _write_json(args.cache_root / "download_result.json", {
        "graphs": args.graph, "failed": failures, "passed": not failures,
    })
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
