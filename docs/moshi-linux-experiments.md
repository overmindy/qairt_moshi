# Moshi Linux Migration and Experiments

## Reproducible real-audio and calibration corpus

Start with Hugging Face's `hf-internal-testing/librispeech_asr_dummy`: its
single `validation` split contains 73 LibriSpeech examples in a 9.19 MB Parquet
file. This is enough for the current real-audio ONNX smoke test and an initial
LM calibration pass; do not download a multi-gigabyte speech corpus yet.

Create ten deterministic 3-second clips. `datasets` downloads the tiny split on
the first run and caches it. The script uses `Audio(decode=False)` to read the
embedded FLAC bytes, then `/usr/bin/ffmpeg` converts them through stdin to
24 kHz mono PCM16. This intentionally bypasses TorchCodec, so Torch 2.6 and its
compatible TorchCodec version do not need to change:

```bash
cd /home/user/yejialei/qairt_moshi
python scripts/moshi_prepare_calibration_audio.py \
  --source hf-dummy \
  --output-dir /data2/yejialei/tmp/moshi-data/tiny-v1 \
  --clean-count 8 --overlap-count 1 --silence-count 1 \
  --seconds 3 --seed 20260915
```

The result contains eight clean clips, one overlap clip, one silence clip,
`smoke.jsonl`, `calibration.jsonl`, and `dataset_summary.json`. The smoke
manifest contains only four clips; use it for the first ONNX comparison. The
ten-clip manifest is deliberately a small initial calibration set, not a claim
of production calibration coverage.

LibriSpeech audio is 16 kHz, so the script resamples it to Moshi's required
24 kHz. That is valid for testing control flow, cache evolution, graph wiring,
and LM calibration from generated codes, but resampling cannot create genuine
8--12 kHz content. Before final quantization of the Mimi encoder/decoder, add a
small number of native 24 kHz recordings (for example LibriTTS clips plus room
noise); the full 7.7 GB archive is still unnecessary unless measurements show
that the small set is insufficient.

## Clean ONNX baseline: complete language-model inference

Start from these three source locations:

- Kyutai's pinned `models/lm.py`: `LMModel.forward()` and
  `forward_depformer_training()` are training paths; `LMGen._step()` is the
  inference call chain that must be preserved.
- `templates/moshi/explicit_cache.py`: converts hidden Python streaming state
  into tensor state. Temporal KV survives between 80 ms frames, while
  DepFormer KV lives for only the eight codebook steps inside one frame.
- `scripts/moshi_export_lm_onnx.py`: validates that rewrite against the pinned
  upstream code before writing any ONNX, exports every LM weight, then executes
  the resulting graphs with ONNX Runtime.

The complete LM is a graph set rather than one `.onnx` file. The 7B FP32 model
exceeds ONNX's single-protobuf size limit, and Qualcomm compilation also needs
smaller partitions. The graph set contains the embedding frontend, all 32
Temporal layers in consecutive shards, the text head, and one statically
specialized graph for each of the eight greedy DepFormer codebook steps.
`manifest.json` records their order, names, cache shapes, and host-owned state.
This is still one complete LM inference step: splitting storage does not omit
weights or change the math. It also keeps each FP32 file below ONNX's 2 GB
protobuf limit; the full DepFormer would exceed that limit because its eight
steps use different attention and feed-forward weights.

Do the cheap, decisive comparison first. It loads the real BF16 checkpoint but
does not spend time writing roughly tens of gigabytes of FP32 ONNX files:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/moshi_streaming_trace.py \
  --model-dir /data2/liuguohong/moshiko-pytorch-bf16 \
  --device cuda:0 --frames 4 --verify-temporal-replay \
  --output-dir /data2/user/moshi-work/moshi-trace-v1

CUDA_VISIBLE_DEVICES=7 python scripts/moshi_export_lm_onnx.py \
  --model-dir /data2/liuguohong/moshiko-pytorch-bf16 \
  --trace-dir /data2/user/moshi-work/moshi-trace-v1 \
  --device cuda:0 --frames 2 --validate-only
```

The required result is `Pinned-upstream inference rewrite parity: PASS`, with
zero error for Temporal hidden/text logits and for DepFormer tokens/logits on
both frames. Only after that passes, export all 32 layers and run ONNX Runtime:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/moshi_export_lm_onnx.py \
  --model-dir /data2/liuguohong/moshiko-pytorch-bf16 \
  --trace-dir /data2/user/moshi-work/moshi-trace-v1 \
  --device cuda:0 --frames 2 --layers-per-shard 2 \
  --output-dir /data2/user/moshi-work/moshi-lm-onnx-v1
```

Use a new empty output directory for every attempt. A successful run requires
all of the following, not merely that files exist: pinned-upstream rewrite
parity, ONNX checker success for every graph, finite ONNX Runtime outputs,
two-frame cache reuse, and per-graph PyTorch/ONNX parity. The Mimi streaming
encoder/decoder are intentionally not included in this LM checkpoint: their
convolution and Transformer streaming states require a separate explicit-state
rewrite before they can be called a correct end-to-end Moshi export.

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

## 7. Isolate Temporal errors before changing quantization

Develop and push code on the Mac; run model, calibration, and cloud-job experiments
from the remote Linux checkout. Do not rerun the already verified FP32 export.
The following probe reuses `source_model_id` (the original ONNX upload, not the
quantized model) from the current manifest. It compiles once without a quantize
job and submits two captured inputs as one inference batch.

```bash
cd /home/user/yejialei/qairt_moshi
git pull --ff-only personal codex/moshi-linux-experiments
export MOSHI_ONNX=/data2/yejialei/tmp/moshi-lm-onnx-v2
export MOSHI_CAL=/data2/yejialei/tmp/moshi-lm-calibration-mp-v1
export MOSHI_BASE=/data2/yejialei/tmp/moshi-lm-dynamic-norm-quantized-30-31-v1
export MOSHI_FLOAT01=/data2/yejialei/tmp/moshi-lm-float-temporal-0-1-v1

/home/user/workdir-wenhao/my_conda_pkgs/qairt/bin/python -u \
  scripts/moshi_probe_float_compile_cloud.py \
  --onnx-dir "$MOSHI_ONNX" --calibration-dir "$MOSHI_CAL" \
  --source-quantization-manifest "$MOSHI_BASE/quantization_manifest.json" \
  --output-dir "$MOSHI_FLOAT01" --graph temporal_layers_0_1 \
  --source-id clean-000 --frame 0 --frames 2 --retry-failed
```

Read `report_frames_2.json`. For each sample, inspect `output_hidden.relative_rms`,
each cache output's `written_slot_metrics` and `preserved_max_abs`, and nonfinite
counts. `input_hidden.fp16_square_inf_count` is a diagnostic for possible FP16
square overflow, not proof of the backend's internal arithmetic. The second
input uses captured FP32 caches; this is **not** a recurrent-chain test. Likewise,
`all_outputs_finite` means no NaN/Inf, not acceptable numerical accuracy.

Compile/inference IDs and output archives are resumable. `--retry-failed` replaces
only failed jobs, not successful jobs with poor numerical results. Default
`--frames 1` retains the existing single-sample report format. Reusing the same
directory for `--frames 2` keeps its successful compile and uses a separate batch
report/archive. Changing graph/source/calibration configuration needs a new
output directory.

### Self-managed quantization versus cloud quantization

Current `scripts/moshi_quantize_lm_graph_set.py` calls `submit_quantize_job` and
then compiles that job's target. A self-managed route instead calibrates on Linux
and exports an AIMET model plus encodings, or a supported ONNX QDQ graph, then
calls `submit_compile_job` directly. The original source upload can be reused
for the float control, but a new locally quantized graph needs its own upload.
No local AIMET quantization implementation is claimed by this probe.

The official [AI Hub compile documentation](https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html)
supports both quantized ONNX and `.aimet` packages. Save the policy, tensor
bitwidth/signedness/per-channel axis, calibration sample IDs, encodings, model
checksums, and tool versions alongside the artifact. Do not request a fresh
`--quantize_full_type` when the intent is to compile existing quantization.
Use `--target_runtime qnn_dlc --truncate_64bit_io` for this graph's int64 I/O.

[AIMET's QDQ conversion documentation](https://qualcomm.github.io/aimet-pages/releases/latest/techniques/onnx_qdq.html)
explains that QDQ stores integer quantization parameters, but plain FP16/BF16
encodings are omitted by that converter. Floating-point exceptions therefore
need explicit representation/verification; converting encodings alone does not
guarantee the intended mixed floating/integer execution. HTP also constrains
valid MatMul input/output type combinations, as the earlier runtime logs showed.

Gate one representative graph in order: float ONNX vs float QNN, locally
quantized ONNX vs float ONNX on identical inputs, quantized QNN vs that local
quantized reference, then a recurrent test. A bad float compiled baseline must
be resolved before attributing all error to quantization or sweeping all shards.

### QAIRT 2.45 reciprocal RMSNorm conversion guard

The downloaded QDQ and rewritten ONNX for `temporal_layers_0_1` agree with
FP32 on CPU (hidden relative RMS approximately 0.0013 and 0.00088), whereas
the previous DLC produced approximately 0.969 and 0.983 on HTP. Inspecting
`OptimizeRMSNormTranslation.match_rms_norm` in SDK
`lib/python/qti/aisw/converters/common/converter_ir/op_graph_optimizations.py`
identified a specific incorrect fusion: its nested
`match_reciprocal_no_affine_transformation` accepts `gamma * reciprocal(rms(x))`
without checking that the multiplication's other input is `x`. It then replaces
the reciprocal numerator with `x`; the real subsequent `Mul(x, factor)` remains,
effectively introducing an extra multiplication by `x`.

`scripts/moshi_guard_qairt_converter.py` inserts an exact input-identity check
before this matcher mutates the graph. A rejected fusion retains primitive ops;
it does not remove QDQ or turn all weights into float. The patch is confined to
the converter process, and rebinds the SDK's cached translation instances as well
as the class method. It fails closed if the expected SDK method structure changes
and records source/patched method hashes. Installed SDK files are untouched.

Use isolated NumPy 1.26.4: this SDK's conversion with the environment's NumPy
2.2.6 also exhibited incorrect reduction axes. Do not downgrade the shared
environment. For the remote experiment, dependencies are in
`/data2/yejialei/tmp/moshi-qdq-0-1-cpu-check-v1/dlc_fusion_diagnostics/python-deps-numpy126`.

The guarded converter is called with existing rewritten QDQ ONNX, not a fresh
export. Preserve output order and float I/O, but set the HTP `position` ABI to
INT32. **Do not include `position` in `--preserve_io_datatype`** when specifying
`--source_model_input_datatype position int32`; preserving its original INT64
conflicts with the override. Keep `--onnx_skip_simplification` so this diagnostic
does not introduce an additional ONNX rewrite.

`scripts/moshi_probe_local_dlc_cloud.py` uploads the resulting DLC once, validates
two captured frames against FP32 ONNX, and then tests frame 1 using the actual
cloud frame-0 caches. Default gates are hidden relative RMS <= 0.02, written-slot
K/V relative RMS <= 0.05, preserved-cache absolute change <= 0.001, and all finite
outputs. A candidate manifest is emitted only after all gates pass. DLC tensor
enumeration is not runtime order: application inputs are sorted by serialized
tensor ID. This matters because `position` has ID 2 while converted float input
`hidden` has ID 1173. AI Hub does not accept an uploaded DLC as a new compile
input, so this script submits inference directly against the uploaded DLC.

Remote artifacts for this representative fix are under
`/data2/yejialei/tmp/moshi-rmsnorm-guard-0-1-v1`:
`temporal_layers_0_1_htp.dlc`, `full_htp_convert.json`, and `probe/report.json`.
The candidate, if all numerical gates pass, is
`probe/candidate/quantization_manifest.json`; it changes only this shard's model.
Do not interpret a representative two-frame test as full-LM or deployment parity.
Apply the same checkpoints to additional affected shards before claiming that.

Verified remote results (2026-10-09): uploaded guarded DLC `mq9e7ey0m`,
captured-frame HTP job `jgodvo845`, recurrent HTP job `jg9o4kdqg`; both jobs and
all numerical gates passed. The initial direct inference `j56o1rdv5` failed on
input order before computation and is retained in the retry history.

| Check | Previous DLC hidden relative RMS | Guarded DLC hidden relative RMS | Worst written K/V relative RMS |
| --- | ---: | ---: | ---: |
| Captured frame 0 | 0.96918744 | 0.00133127 | 0.02196464 |
| Captured frame 1 | 0.98306137 | 0.00068711 | 0.01406254 |
| Frame 1 with guarded frame-0 caches, FP32 oracle fed the same caches | Not tested | 0.00060810 | 0.01405803 |

All outputs were finite. Maximum preserved-cache change was 0.00010997 in the
captured test and 0.00003464 in the recurrent test. All eight main projection
weights retained their original datatypes and quantization parameters (five
INT8, three FP16). The candidate differs from the baseline only in
`temporal_layers_0_1`, and `_compiled_input_order` successfully read and verified
its runtime order through the completed inference receipt. Tiny extracted
RMSNorm CPU validation also passed (max absolute error 2.98e-8). Eight focused
regression tests passed on remote Python 3.10.
