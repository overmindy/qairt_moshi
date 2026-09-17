"""Summarize completed per-stage errors from Moshi chained cloud validation."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.8g}"


def _metric_summary(metrics: list[dict[str, float]]) -> dict[str, float] | None:
    if not metrics:
        return None
    return {
        "maximum_relative_rms": max(metric["relative_rms"] for metric in metrics),
        "mean_relative_rms": statistics.fmean(
            metric["relative_rms"] for metric in metrics
        ),
        "maximum_rmse": max(metric["rmse"] for metric in metrics),
        "maximum_abs": max(metric["max_abs"] for metric in metrics),
    }


def _temporal_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("kind") == "temporal":
            by_graph[record["graph"]].append(record)

    graphs = []
    for graph, graph_records in by_graph.items():
        cache_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in graph_records:
            for name, cache in record.get("cache", {}).items():
                cache_by_name[name].append(cache)
        cache_summaries = []
        for name, caches in cache_by_name.items():
            written = _metric_summary([cache["written"] for cache in caches])
            assert written is not None
            cache_summaries.append(
                {
                    "name": name,
                    "written": written,
                    "maximum_preserved_abs": max(
                        cache["preserved_max_abs"] for cache in caches
                    ),
                }
            )
        cache_summaries.sort(
            key=lambda item: item["written"]["maximum_relative_rms"], reverse=True
        )
        hidden = _metric_summary([record["hidden"] for record in graph_records])
        assert hidden is not None
        graphs.append(
            {
                "graph": graph,
                "completed_records": len(graph_records),
                "sources": sorted({record["source_id"] for record in graph_records}),
                "frames": sorted({record["frame"] for record in graph_records}),
                "hidden": hidden,
                "caches": cache_summaries,
                "records": [
                    {
                        "source_id": record["source_id"],
                        "frame": record["frame"],
                        "hidden": record["hidden"],
                        "cache": record.get("cache", {}),
                    }
                    for record in graph_records
                ],
            }
        )
    graphs.sort(key=lambda item: item["hidden"]["maximum_relative_rms"], reverse=True)
    return {
        "ranked_by_hidden_relative_rms": graphs,
        "execution_order": list(by_graph),
    }


def _other_stage_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for kind in ("frontend", "head", "depformer"):
        stage_records = [record for record in records if record.get("kind") == kind]
        if not stage_records:
            continue
        metric_fields = {
            name
            for record in stage_records
            for name, value in record.items()
            if isinstance(value, dict) and "relative_rms" in value
        }
        summary = {
            "kind": kind,
            "completed_records": len(stage_records),
            "graphs": list(dict.fromkeys(record["graph"] for record in stage_records)),
            "metrics": {
                field: _metric_summary(
                    [record[field] for record in stage_records if field in record]
                )
                for field in sorted(metric_fields)
            },
        }
        token_matches = [
            record["token_match"] for record in stage_records if "token_match" in record
        ]
        if token_matches:
            summary["token_agreement"] = statistics.fmean(token_matches)
        summaries.append(summary)
    return summaries


def summarize(
    records: list[dict[str, Any]], expected_graphs: list[str] | None = None
) -> dict[str, Any]:
    temporal = _temporal_summary(records)
    completed_graphs = list(
        dict.fromkeys(record["graph"] for record in records if "graph" in record)
    )
    result = {
        "format": "moshi-lm-stage-error-summary-v1",
        "completed_record_count": len(records),
        "completed_graphs_in_order": completed_graphs,
        "temporal": temporal,
        "other_stages": _other_stage_summary(records),
    }
    if expected_graphs is not None:
        result["expected_graphs_in_order"] = expected_graphs
        result["missing_expected_graphs"] = [
            graph for graph in expected_graphs if graph not in completed_graphs
        ]
    return result


def _expected_graphs(onnx_dir: Path) -> list[str]:
    manifest_path = onnx_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing ONNX graph manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    return [
        "frontend",
        *(Path(item["onnx"]).stem for item in manifest["temporal_shards"]),
        "head",
        *(Path(item["onnx"]).stem for item in manifest["depformer"]["steps"]),
    ]


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Moshi chained stage error summary",
        "",
        "Completed graphs in execution order: "
        + ", ".join(f"`{name}`" for name in summary["completed_graphs_in_order"]),
        "",
    ]
    if "missing_expected_graphs" in summary:
        missing = summary["missing_expected_graphs"]
        lines.extend(
            [
                "Missing expected graphs (no completed metric, not a pass): "
                + (", ".join(f"`{name}`" for name in missing) if missing else "none"),
                "",
            ]
        )
    lines.extend(
        [
            "## Temporal shards ranked by shard-output hidden error",
            "",
            "Each Temporal graph contains two layers. `hidden` is measured after both layers; the named K/V outputs identify the individual layer that wrote each cache.",
            "",
            "| Rank | Graph | Records | Frames | Hidden max rel RMS | Hidden mean rel RMS | Hidden max RMSE | Hidden max abs | Worst cache | Cache max rel RMS | Preserved max abs |",
            "|---:|---|---:|---|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    graphs = summary["temporal"]["ranked_by_hidden_relative_rms"]
    for rank, graph in enumerate(graphs, start=1):
        hidden = graph["hidden"]
        cache = graph["caches"][0] if graph["caches"] else None
        lines.append(
            "| {rank} | {graph} | {records} | {frames} | {hidden_max} | "
            "{hidden_mean} | {hidden_rmse} | {hidden_abs} | {cache_name} | "
            "{cache_rel} | {preserved} |".format(
                rank=rank,
                graph=graph["graph"],
                records=graph["completed_records"],
                frames=",".join(str(frame) for frame in graph["frames"]),
                hidden_max=_number(hidden["maximum_relative_rms"]),
                hidden_mean=_number(hidden["mean_relative_rms"]),
                hidden_rmse=_number(hidden["maximum_rmse"]),
                hidden_abs=_number(hidden["maximum_abs"]),
                cache_name=cache["name"] if cache else "-",
                cache_rel=(
                    _number(cache["written"]["maximum_relative_rms"]) if cache else "-"
                ),
                preserved=(_number(cache["maximum_preserved_abs"]) if cache else "-"),
            )
        )

    lines.extend(["", "## Cache outputs ranked within each shard", ""])
    for graph in graphs:
        lines.extend(
            [
                f"### {graph['graph']}",
                "",
                "| Cache output | Written max rel RMS | Written mean rel RMS | Written max RMSE | Written max abs | Preserved max abs |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for cache in graph["caches"]:
            written = cache["written"]
            lines.append(
                "| {name} | {max_rel} | {mean_rel} | {rmse} | {max_abs} | "
                "{preserved} |".format(
                    name=cache["name"],
                    max_rel=_number(written["maximum_relative_rms"]),
                    mean_rel=_number(written["mean_relative_rms"]),
                    rmse=_number(written["maximum_rmse"]),
                    max_abs=_number(written["maximum_abs"]),
                    preserved=_number(cache["maximum_preserved_abs"]),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Other completed stages",
            "",
            "| Stage | Graphs | Records | Output | Max rel RMS | Mean rel RMS | Max RMSE | Max abs | Token agreement |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in summary["other_stages"]:
        token_agreement = stage.get("token_agreement")
        for output_name, metrics in stage["metrics"].items():
            assert metrics is not None
            lines.append(
                "| {stage} | {graphs} | {records} | {output} | {max_rel} | "
                "{mean_rel} | {rmse} | {max_abs} | {tokens} |".format(
                    stage=stage["kind"],
                    graphs=", ".join(stage["graphs"]),
                    records=stage["completed_records"],
                    output=output_name,
                    max_rel=_number(metrics["maximum_relative_rms"]),
                    mean_rel=_number(metrics["mean_relative_rms"]),
                    rmse=_number(metrics["maximum_rmse"]),
                    max_abs=_number(metrics["maximum_abs"]),
                    tokens=_number(token_agreement),
                )
            )
    lines.append("")

    lines.extend(["## Per-frame temporal hidden errors", ""])
    for graph in graphs:
        lines.extend(
            [
                f"### {graph['graph']}",
                "",
                "| Source | Frame | Relative RMS | RMSE | Max abs |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for record in sorted(
            graph["records"], key=lambda item: (item["frame"], item["source_id"])
        ):
            hidden = record["hidden"]
            lines.append(
                "| {source} | {frame} | {relative} | {rmse} | {max_abs} |".format(
                    source=record["source_id"],
                    frame=record["frame"],
                    relative=_number(hidden["relative_rms"]),
                    rmse=_number(hidden["rmse"]),
                    max_abs=_number(hidden["max_abs"]),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        help="Optional graph set used to report expected graphs that have no metrics",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    metrics_path = args.validation_dir / "stage_metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(f"Missing stage metrics: {metrics_path}")
    records = json.loads(metrics_path.read_text())
    if not isinstance(records, list):
        raise SystemExit(f"Expected a list in {metrics_path}")
    output_dir = args.output_dir or args.validation_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_graphs = _expected_graphs(args.onnx_dir) if args.onnx_dir else None
    summary = summarize(records, expected_graphs)
    json_path = output_dir / "stage_summary.json"
    markdown_path = output_dir / "STAGE_SUMMARY.md"
    _write_json(json_path, summary)
    _write_markdown(markdown_path, summary)
    print(markdown_path.read_text(), end="")
    print(f"JSON summary: {json_path}")


if __name__ == "__main__":
    main()
