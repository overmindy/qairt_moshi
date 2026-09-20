"""The codebook-0 attention shortcut must preserve the real block contract."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qai_hub_models.models.templates.moshi.explicit_cache import DepFormerStepBlock


def _block_source(kv_repeat: int = 1) -> SimpleNamespace:
    attention = SimpleNamespace(
        weights_per_step_schedule=None,
        context=4,
        weights_per_step=None,
        num_heads=2,
        kv_repeat=kv_repeat,
        embed_dim=4,
        in_projs=nn.ModuleList([nn.Linear(4, 4 + 2 * (4 // kv_repeat))]),
        out_projs=nn.ModuleList([nn.Linear(4, 4)]),
    )
    return SimpleNamespace(
        self_attn=attention,
        norm1=nn.Identity(),
        norm2=nn.Identity(),
        layer_scale_1=nn.Identity(),
        layer_scale_2=nn.Identity(),
        activation=nn.Tanh(),
        linear1=nn.Linear(4, 4),
        linear2=nn.Linear(4, 4),
        gating=None,
    )


@pytest.mark.parametrize("kv_repeat", [1, 2])
def test_codebook_zero_single_value_matches_masked_attention(kv_repeat: int) -> None:
    torch.manual_seed(7)
    layer = _block_source(kv_repeat)
    standard = DepFormerStepBlock(layer, 0)
    shortcut = DepFormerStepBlock(layer, 0, single_value_attention=True)
    hidden = torch.randn(1, 1, 4)
    key_cache = torch.randn(1, 2 // kv_repeat, 4, 2)
    value_cache = torch.randn(1, 2 // kv_repeat, 4, 2)

    expected = standard(hidden, key_cache, value_cache)
    actual = shortcut(hidden, key_cache, value_cache)

    for received, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(received, reference, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual[1][:, :, 0], expected[1][:, :, 0])
    torch.testing.assert_close(actual[2][:, :, 0], expected[2][:, :, 0])


def test_single_value_rejects_later_codebooks() -> None:
    with pytest.raises(ValueError, match="only valid for codebook 0"):
        DepFormerStepBlock(_block_source(), 1, single_value_attention=True)


@pytest.mark.parametrize("kv_repeat", [1, 2])
def test_single_value_onnx_omits_attention_softmax(tmp_path, kv_repeat: int) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(11)
    layer = _block_source(kv_repeat)
    standard = DepFormerStepBlock(layer, 0).eval()
    shortcut = DepFormerStepBlock(layer, 0, single_value_attention=True).eval()
    inputs = (
        torch.randn(1, 1, 4),
        torch.zeros(1, 2 // kv_repeat, 4, 2),
        torch.zeros(1, 2 // kv_repeat, 4, 2),
    )
    path = tmp_path / "depformer_block0_single_value.onnx"
    torch.onnx.export(
        shortcut,
        inputs,
        str(path),
        opset_version=17,
        dynamo=False,
        input_names=["hidden", "key_cache", "value_cache"],
        output_names=["output_hidden", "output_key", "output_value"],
    )

    assert all(node.op_type != "Softmax" for node in onnx.load(path).graph.node)
    expected = standard(*inputs)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, dict(zip(
        ["hidden", "key_cache", "value_cache"],
        [value.numpy() for value in inputs],
        strict=True,
    )))
    for received, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(torch.from_numpy(received), reference, rtol=0, atol=1e-6)
