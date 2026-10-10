"""Intermediate probes must not reinterpret position/index arithmetic as FLOAT."""
import importlib.util
from pathlib import Path
import sys
import unittest

import onnx
from onnx import helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from moshi_compare_qdq_stages_cpu import inferred_values, floating_value


class StageTypesTest(unittest.TestCase):
    def test_integer_position_is_excluded_and_fp16_preserved(self):
        graph = helper.make_graph([
            helper.make_node("Add", ["position", "one"], ["next_position"], name="/Add"),
            helper.make_node("MatMul", ["x", "w"], ["hidden"], name="/MatMul"),
        ], "types", [helper.make_tensor_value_info("position", TensorProto.INT64, [1]),
                     helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 2])],
            [helper.make_tensor_value_info("hidden", TensorProto.FLOAT16, [1, 2])],
            [helper.make_tensor("one", TensorProto.INT64, [1], [1]),
             helper.make_tensor("w", TensorProto.FLOAT16, [2, 2], [1, 0, 0, 1])])
        values = inferred_values(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]))
        self.assertFalse(floating_value(values, "next_position"))
        self.assertTrue(floating_value(values, "hidden"))
        self.assertEqual(values["hidden"].type.tensor_type.elem_type, TensorProto.FLOAT16)

    def test_unknown_type_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Cannot infer"):
            floating_value({}, "unknown")


if __name__ == "__main__":
    unittest.main()
