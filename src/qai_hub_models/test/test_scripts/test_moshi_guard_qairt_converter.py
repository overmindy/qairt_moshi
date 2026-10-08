"""The SDK guard must reject gamma without mutating the graph."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/moshi_guard_qairt_converter.py"
spec = importlib.util.spec_from_file_location("moshi_guard", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

SOURCE = '''
def match_rms_norm(self, graph):
    def match_reciprocal_no_affine_transformation(node_tuple):
        first_node = node_tuple[0]
        div_node = node_tuple[4]
        second_mul_node = node_tuple[5]
        graph.mutated = True
        return "fused"
    return match_reciprocal_no_affine_transformation(graph.nodes)
'''


@pytest.mark.parametrize("other_input,expected", [("x", "fused"), ("gamma", None), ("cast_x", None)])
def test_guard_before_mutation(other_input, expected):
    node = lambda inputs, outputs: SimpleNamespace(input_names=inputs, output_names=outputs)
    graph = SimpleNamespace(mutated=False, nodes=[node(["x", "x"], ["square"]), None, None,
                                                 None, node(["one", "rms"], ["reciprocal"]),
                                                 node([other_input, "reciprocal"], ["factor"])])
    namespace = {}
    exec(module.guarded_method_source(SOURCE), namespace)
    assert namespace["match_rms_norm"](None, graph) == expected
    assert graph.mutated == (expected == "fused")


def test_unknown_sdk_fails_closed():
    with pytest.raises(ValueError, match="binding changed"):
        module.guarded_method_source(SOURCE.replace("node_tuple[5]", "node_tuple[6]"))
