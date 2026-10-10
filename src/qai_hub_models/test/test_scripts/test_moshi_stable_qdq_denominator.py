"""Denominator rewrite must preserve RMS units and existing QDQ nodes."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from moshi_patch_existing_qdq_rmsnorm import stabilize_denominator, remove_norm_activation_qdq


class TestDenominator(unittest.TestCase):
    def test_explicit_output_refinement_preserves_output_abi(self):
        graph = helper.make_graph([
            helper.make_node("Identity", ["x"], ["residual"]),
            helper.make_node("QuantizeLinear", ["residual", "s", "z"], ["q"], name="q"),
            helper.make_node("DequantizeLinear", ["q", "s", "z"], ["out"], name="dq"),
        ], "output", [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("out", onnx.TensorProto.FLOAT, [1])],
            [numpy_helper.from_array(np.array(0.1, np.float32), "s"),
             numpy_helper.from_array(np.array(0, np.uint8), "z")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
        self.assertEqual(remove_norm_activation_qdq(model, ["residual"]), ["dq", "q"])
        onnx.checker.check_model(model)
        actual = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
            None, {"x": np.array([0.1234], np.float32)})[0]
        np.testing.assert_array_equal(actual, np.array([0.1234], np.float32))
        self.assertEqual(model.graph.output[0].name, "out")
        with self.assertRaisesRegex(ValueError, "Projection weights"):
            remove_norm_activation_qdq(model, ["onnx::MatMul_1"])

    def test_norm_activation_removal_keeps_projection_weight_quantizers(self):
        nodes = [
            helper.make_node("Identity", ["x"], ["/norm2/Sqrt_output_0"], name="norm"),
            helper.make_node("QuantizeLinear", ["/norm2/Sqrt_output_0", "s", "z"], ["nq"], name="norm_q"),
            helper.make_node("DequantizeLinear", ["nq", "s", "z"], ["ndq"], name="norm_dq"),
            helper.make_node("QuantizeLinear", ["onnx::MatMul_1", "s", "z"], ["wq"], name="weight_q"),
            helper.make_node("DequantizeLinear", ["wq", "s", "z"], ["wdq"], name="weight_dq"),
            helper.make_node("MatMul", ["ndq", "wdq"], ["out"], name="projection"),
        ]
        graph = helper.make_graph(nodes, "removal", [], [])
        model = helper.make_model(graph)
        before = {n.name: n.SerializeToString() for n in model.graph.node if n.name.startswith("weight_")}
        removed = remove_norm_activation_qdq(model)
        self.assertEqual(removed, ["norm_dq", "norm_q"])
        for node in model.graph.node:
            if node.name in before:
                self.assertEqual(node.SerializeToString(), before[node.name])
        projection = next(n for n in model.graph.node if n.name == "projection")
        self.assertEqual(list(projection.input), ["/norm2/Sqrt_output_0", "wdq"])

    def test_restores_units_before_quantizer_and_keeps_encodings(self):
        nodes = [
            helper.make_node("Pow", ["x", "two"], ["square"], name="/norm1_1/Pow"),
            helper.make_node("ReduceMean", ["square"], ["mean"], axes=[-1], keepdims=1, name="/norm1_1/ReduceMean"),
            helper.make_node("Add", ["mean", "epsilon"], ["variance"], name="/norm1_1/Add"),
            helper.make_node("Sqrt", ["variance"], ["rms"], name="/norm1_1/Sqrt"),
            helper.make_node("QuantizeLinear", ["rms", "scale", "zero"], ["q"], name="existing_quantizer"),
            helper.make_node("DequantizeLinear", ["q", "scale", "zero"], ["dq"], name="existing_dequantizer"),
        ]
        constants = [numpy_helper.from_array(np.array(v, dtype=dtype), name)
                     for name, v, dtype in (("two", 2, np.float32), ("epsilon", 1e-8, np.float32),
                                            ("scale", 1, np.float32), ("zero", 0, np.uint8))]
        graph = helper.make_graph(nodes, "norm", [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])],
                                  [helper.make_tensor_value_info("rms", onnx.TensorProto.FLOAT, [1, 1]),
                                   helper.make_tensor_value_info("dq", onnx.TensorProto.FLOAT, [1, 1])], constants)
        original = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
        patched = copy.deepcopy(original)
        stabilize_denominator(patched, "/norm1_1/")
        before = [n.SerializeToString() for n in original.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
        after = [n.SerializeToString() for n in patched.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear")]
        self.assertEqual(before, after)
        for source in original.graph.initializer:
            target = next(v for v in patched.graph.initializer if v.name == source.name)
            np.testing.assert_array_equal(numpy_helper.to_array(source), numpy_helper.to_array(target))
        feed = {"x": np.array([[290, 1, -2, 0.2]], np.float32)}
        expected = ort.InferenceSession(original.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feed)
        actual = ort.InferenceSession(patched.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feed)
        np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6)
        np.testing.assert_array_equal(actual[1], expected[1])


if __name__ == "__main__":
    unittest.main()
