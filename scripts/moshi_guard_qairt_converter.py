"""Run QAIRT with a conservative guard for its reciprocal RMSNorm matcher.

QAIRT 2.45's match_reciprocal_no_affine_transformation can mistake
gamma * reciprocal(rms(x)) for x * reciprocal(rms(x)). Reject that match
before any graph mutation. Source ONNX/QDQ, encodings, and installed SDK
files are untouched. Only this converter process uses the patched method.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap
from typing import Any

TARGET = "match_reciprocal_no_affine_transformation"
GUARD = """
moshi_other_inputs = [name for name in second_mul_node.input_names
                      if name != div_node.output_names[0]]
if (len(moshi_other_inputs) != 1
        or moshi_other_inputs[0] != first_node.input_names[0]):
    return None
"""


def guarded_method_source(source: str) -> str:
    """Fail closed if the expected SDK method structure has changed."""
    tree = ast.parse(textwrap.dedent(source))
    targets = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == TARGET]
    if len(targets) != 1:
        raise ValueError("SDK reciprocal RMSNorm matcher changed; refusing to patch")
    target = targets[0]
    matches = [i for i, n in enumerate(target.body)
               if isinstance(n, ast.Assign) and len(n.targets) == 1
               and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id == "second_mul_node"
               and ast.unparse(n.value) == "node_tuple[5]"]
    if len(matches) != 1:
        raise ValueError("SDK second multiplication binding changed; refusing to patch")
    index = matches[0]
    bindings = {n.targets[0].id for n in target.body[:index]
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)}
    if not {"first_node", "div_node"}.issubset(bindings):
        raise ValueError("SDK RMSNorm input bindings changed; refusing to patch")
    # Conservative: casts not referring to the exact same input also fall back
    # to primitive ops. That sacrifices a possible fusion, not correctness.
    target.body[index + 1:index + 1] = ast.parse(GUARD).body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def install_guard(module: Any) -> dict[str, str]:
    cls = module.OptimizeRMSNormTranslation
    original = cls.match_rms_norm
    source = inspect.getsource(original)
    patched = guarded_method_source(source)
    namespace: dict[str, Any] = {}
    exec(compile(patched, "<moshi-guarded-rmsnorm>", "exec"), original.__globals__, namespace)
    replacement = namespace[original.__name__]
    # Preserve the public method identity used by the optimization registry.
    replacement.__module__ = original.__module__
    replacement.__qualname__ = original.__qualname__
    cls.match_rms_norm = replacement
    # The SDK decorator creates translation instances during import and caches
    # bound methods in a registry. Updating the class alone leaves those cached
    # methods pointing at the original code. Rebind every live RMSNorm entry.
    instances = {id(value): value for value in module.OptimizationTranslations.translations.values()
                 if isinstance(value, cls)}
    if not instances:
        raise ValueError("SDK RMSNorm translation registry changed; refusing to convert")
    for instance in instances.values():
        instance.register_method(module.MATCH_RMSNORM, replacement.__get__(instance, cls))
    return {"target": TARGET,
            "registry_instances": str(len(instances)),
            "sdk_method_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "guarded_method_sha256": hashlib.sha256(patched.encode()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--numpy-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("converter_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sdk = args.sdk_root.resolve()
    tool = sdk / "bin/x86_64-linux-clang/qairt-converter"
    if not tool.is_file() or not (args.numpy_dir / "numpy").is_dir():
        raise ValueError("SDK or isolated NumPy directory is missing")
    if args.receipt.exists():
        raise ValueError("Conversion receipt exists; use a new output/receipt path")
    converter_args = args.converter_args
    if converter_args[:1] == ["--"]:
        converter_args = converter_args[1:]
    if not converter_args:
        parser.error("Supply converter arguments after --")
    sys.path[:0] = [str(args.numpy_dir.resolve()), str(sdk / "lib/python")]
    os.environ["QAIRT_SDK_ROOT"] = str(sdk)
    import numpy as np
    if np.__version__ != "1.26.4":
        raise ValueError(f"Expected isolated NumPy 1.26.4; loaded {np.__version__}")
    from qti.aisw.converters.common.converter_ir import op_graph_optimizations
    guard = install_guard(op_graph_optimizations)
    receipt = {"format": "moshi-qairt-rmsnorm-guard-v1", "sdk_root": str(sdk),
               "numpy_version": np.__version__, "converter_args": converter_args,
               "guard": guard, "status": "running"}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Installed process-local RMSNorm input-identity guard", json.dumps(guard), flush=True)
    sys.argv = [str(tool), *converter_args]
    try:
        try:
            runpy.run_path(str(tool), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        receipt["status"] = "success"
    except BaseException:
        receipt["status"] = "failed"
        raise
    finally:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
