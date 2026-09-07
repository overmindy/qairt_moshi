# Moshi Linux Migration and Experiments

## Explicit Temporal cache and first compiler artifact

After collecting `temporal_sequence.pt`, run:

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/moshi_export_temporal.py \
  --model-dir /data2/liuguohong/moshiko-pytorch-bf16 \
  --trace-dir /tmp/moshi-golden-replay --device cuda:0 \
  --output-dir /tmp/moshi-temporal-export
```

Requires PyTorch and ONNX in the checkpoint environment. The implementation
supports batch one, one token per call, unconditional Temporal layers and the
original full cache capacity. Each layer receives `[2,1,KV_heads,capacity,head_dim]`
K/V storage. `position` is an int64 tensor, drives RoPE and ring writes, and is
returned incremented by the complete Temporal wrapper. No streaming contexts
or Python cache objects are used by this wrapper's forward computation.

The command checks all Temporal hidden states and text logits against the saved
BF16 golden trace with zero tolerance. On mismatch it stops before export.
On success it copies only the real first block to CPU FP32, exports
`temporal_block_0.onnx` (opset 17), runs the ONNX checker, and saves
`block_0_reference.pt`. This block includes normalization, QKV, RoPE, functional
ring cache writes, attention, residuals and the feedforward network. It is a
compiler compatibility probe, not a full Moshi model. FP32 ONNX runtime parity,
cache wraparound parity, QAIRT conversion, HTP precision selection and target
compilation remain unverified. The int64 position and ScatterElements operations
may require converter-specific lowering. SDK version and target SoC must be
established before selecting the local QAIRT conversion/compilation commands.

This repository contains the Moshi recipe, but it does not contain the 18 GB Hugging Face checkpoint. Do not commit model weights, Hugging Face cache directories, generated binaries, or WAV files.

## 1. Push the branch from macOS

The current branch is intended to be pushed as `codex/moshi-linux-experiments`.

```bash
git switch -c codex/moshi-linux-experiments
git add src/qai_hub_models/models/moshi src/qai_hub_models/models/templates/moshi src/qai_hub_models/scorecard/models/moshi scripts/moshi_experiment.py docs/moshi-linux-experiments.md
git commit -m "Add real Moshi recipe and Linux experiment entrypoint"
git remote add personal https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u personal codex/moshi-linux-experiments
```

If `personal` already exists, update it instead of adding it again:

```bash
git remote set-url personal https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u personal codex/moshi-linux-experiments
```

The existing `origin` points to the Qualcomm upstream repository. Do not push this branch to `origin` unless you explicitly intend to create an upstream contribution.

## 2. Clone on Linux

```bash
git clone --branch codex/moshi-linux-experiments https://github.com/<YOUR_USER>/<YOUR_REPO>.git
cd ai-hub-models
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r src/qai_hub_models/models/moshi/requirements.txt
```

Keep the Hugging Face cache on a disk with at least 30 GB free:

```bash
export HF_HOME=/data/$USER/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
df -h "$HF_HOME"
```

## 3. Checks that do not load weights

Run these first. They do not call `from_pretrained()`:

```bash
python scripts/moshi_experiment.py
python -m py_compile src/qai_hub_models/models/moshi/*.py src/qai_hub_models/models/templates/moshi/*.py
git diff --check
```

## 4. Download and real one-frame smoke

Only run this on a Linux host with enough RAM and swap. It loads the real Moshiko BF16 checkpoint and runs one 1920-sample frame:

```bash
python scripts/moshi_experiment.py --run-real --hf-repo kyutai/moshiko-pytorch-bf16
```

For an already downloaded local snapshot, specify both its directory and GPU:

```bash
python scripts/moshi_experiment.py \
  --run-real \
  --model-dir /data/models/moshiko-pytorch-bf16 \
  --device cuda:0 \
  --dtype bfloat16
```

The directory must contain `model.safetensors`, `tokenizer-e351c8d8-checkpoint125.safetensors`, and `tokenizer_spm_32k_3.model`. If it contains `config.json`, any filenames referenced by that configuration must also exist in the directory. Local-directory mode does not download missing files.

To expose only one physical GPU to the process, use `CUDA_VISIBLE_DEVICES`. The visible GPU is then numbered from zero inside the process:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/moshi_experiment.py \
  --run-real \
  --model-dir /data/models/moshiko-pytorch-bf16 \
  --device cuda:0 \
  --dtype bfloat16
```

Use one process, no pytest-xdist, and no parallel export jobs. Capture the complete output:

```bash
python scripts/moshi_experiment.py --run-real --hf-repo kyutai/moshiko-pytorch-bf16 2>&1 | tee moshi-real-smoke.log
```

## 5. Experiment order

1. Static checks and import checks.
2. Real checkpoint load only; record peak RSS and available RAM.
3. One-frame PyTorch smoke; verify waveform and text-logit shapes.
4. Golden trace: save input audio, Mimi codes, Temporal hidden state, DepFormer logits, sampled codebooks, and reconstructed WAV.
5. Export each component separately with float precision; do not export the whole autoregressive loop as one graph.
6. Compare exported outputs with the golden trace before any quantization.
7. Try W8A16/W4A16 only after FP parity; calibrate with real audio-derived codebook states.
8. Compile/profile on the target Snapdragon device and record placement, HTP core use, latency, peak memory, token parity, and WAV validity.

The current recipe is not evidence that steps 4–8 have passed. In particular, it is not yet a complete streaming KV-cache runtime.

### Multi-frame golden trace

Run the real checkpoint with greedy generation and four silence frames:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/moshi_streaming_trace.py \
  --model-dir /data2/liuguohong/moshiko-pytorch-bf16 \
  --device cuda:0 --frames 4 --output-dir /tmp/moshi-golden
```

This is a structural streaming baseline, not an audio quality test or calibration
dataset. Each input frame is 1920 mono samples at 24 kHz. Mimi and LMGen retain
their streaming state throughout the run. The driver uses the pinned upstream
LMGen delay handling and sequential DepFormer sampling. Only the eight audio
channels of LMGen output are passed to Mimi; the leading text channel is saved
separately. The driver requires more input frames than LMGen's maximum delay.
It does not flush pending delayed frames after the input ends.

The output directory contains input_audio.pt, waveform.wav, and a tensor file
for every trace key:

| Key | Layout | Time axis |
| --- | --- | --- |
| mimi_codes | B, 8, T | Input frames |
| temporal_sequence | B, 17, T | Actual delay-processed LM inputs, including initial tokens |
| temporal_hidden | B, T, D | Every LM step, including warmup |
| text_logits | B, 1, T, text vocabulary | Every LM step, including warmup |
| depformer_logits | B, T, 8, 1, audio vocabulary | Every LM step, sequential codebook order |
| text_tokens | B, 1, Tout | Delay-aligned output |
| audio_codes | B, 8, Tout | Delay-aligned output |
| waveform | B, 1, samples | Decoded output |

Here Tout = T - max_delay for the default synchronous LMGen. The hidden state
and logits at step t must not be naively paired with output frame t: LMGen
applies codebook-specific delays before returning output. Trace collection
deliberately bypasses CUDA Graph wrappers so Python tensor capture executes on
every step. These runs measure correctness, not production inference latency.
Real checkpoint execution of this new entrypoint still needs Linux validation.

Add `--verify-temporal-replay` to independently replay the saved Temporal inputs
through `forward_text` in a fresh upstream streaming context, reusing the loaded
weights. This checks exact hidden and text-logit equality for every frame and
fails on any mismatch. It supports unconditioned checkpoints only. This is a
module-boundary and state-reset check, not an explicit-cache implementation or
an export parity test. The new sequence artifact includes the delayed text,
generated audio, and user audio streams; Mimi codes alone cannot reconstruct it.

## 6. Updating the branch

```bash
git pull --ff-only personal codex/moshi-linux-experiments
git status --short
git add <changed-files>
git commit -m "Describe the experiment change"
git push
```

Never add `~/.cache/huggingface`, `$HF_HOME`, `.venv`, `build/`, `*.safetensors`, `*.pt`, `*.wav`, or exported QNN binaries to Git.
