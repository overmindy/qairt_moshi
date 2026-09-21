# Mimi float codec on Android (SM8850 experiment)

This directory contains a small native host for the **explicit-state** Mimi
Encoder and Decoder DLCs. It reconstructs audio without the Moshi LM:

`24 kHz mono PCM16 WAV -> 1920-sample frame -> QNN Encoder -> 8 codes -> QNN Decoder -> WAV`

Start with `src/qai_hub_models/models/templates/moshi/model.py` for the static
wrappers, `app.py` for the real `mimi.streaming(...)` call path, and
`explicit_mimi.py` for the exported position, KV-cache, and convolution-state
contract. The static wrappers alone do not implement continuous audio.

## Artifacts and requirements

Use the float `qnn_dlc` models compiled from the explicit-state ONNX v2 graphs:

| Component | AI Hub compiled model ID | Graph data input | Graph data output |
| --- | --- | --- | --- |
| Encoder | `mnwvyowkq` | `audio`: float32 `[1,1,1920]` | `output_0`: int32 `[1,8,1]` |
| Decoder | `mqeyoy77m` | `codes`: int32 `[1,8,1]` | `output_0`: float32 `[1,1,1920]` |

Each graph also receives `position` int32 `[1]`, `kv_cache` float32
`[16,1,8,250,64]`, and `conv_state` float32 `[1,11526]` (Encoder) or
`[1,12672]` (Decoder). Outputs `output_1`, `output_2`, and `output_3`
carry the corresponding next state. The host initializes these to zero,
feeds each component's state back into itself, checks that position advances
by two per 80 ms frame, and resets the states on each new process run.

The tested phone was BKQ-AN20, SM8850, Android 16, arm64-v8a, serial
`ATWV026602002286`. The tested QAIRT version was `2.45.0.260326`; the
models were compiled with `--target_runtime qnn_dlc --truncate_64bit_io`.
The SDK headers and runtime libraries are external dependencies; they are
not included in this repository. The Android NDK is needed to build the host.
These DLCs do **not** need conversion to a Genie `.bin`. The present GenieX
LLM/VLM plugin does not supply this codec host loop.

## Build and run

From the repository root, set the installed SDK and NDK locations:

```sh
QNN_INCLUDE=/path/to/qairt/qnn-api/include
NDK=/path/to/android-ndk
"$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android29-clang++" \
  -std=c++17 -O2 -Wall -Wextra -Werror -static-libstdc++ \
  -I "$QNN_INCLUDE" android/mimi_qnn_runner.cpp \
  -o mimi_qnn_runner -ldl
```

On Linux, replace `darwin-x86_64` with `linux-x86_64`. The phone needs matching
QAIRT HTP libraries, including `libQnnHtp.so`, `libQnnSystem.so`,
`libQnnHtpV81Stub.so`, and `libQnnHtpV81Skel.so`. This experiment reused
`/data/local/tmp/geniex/lib/qairt/htp-files` from an existing GenieX install.
Use a phone-specific path if its layout differs.
Place the two downloaded DLCs and a 24 kHz mono PCM16 `input.wav` in the
current directory before running the commands below, or replace their paths.

```sh
SERIAL=ATWV026602002286
PHONE_DIR=/data/local/tmp/moshi-mimi-codec-v1
QNN_LIB_DIR=/data/local/tmp/geniex/lib/qairt/htp-files
adb -s "$SERIAL" shell "mkdir -p $PHONE_DIR"
adb -s "$SERIAL" push mimi_qnn_runner "$PHONE_DIR/"
adb -s "$SERIAL" push mimi_encoder.dlc "$PHONE_DIR/"
adb -s "$SERIAL" push mimi_decoder.dlc "$PHONE_DIR/"
adb -s "$SERIAL" push input.wav "$PHONE_DIR/"
adb -s "$SERIAL" shell "cd $PHONE_DIR && LD_LIBRARY_PATH=$QNN_LIB_DIR ADSP_LIBRARY_PATH=$QNN_LIB_DIR ./mimi_qnn_runner $QNN_LIB_DIR mimi_encoder.dlc mimi_decoder.dlc input.wav output.wav 38 > run.log 2>&1"
adb -s "$SERIAL" pull "$PHONE_DIR/output.wav" .
adb -s "$SERIAL" pull "$PHONE_DIR/output.wav.codes.txt" .
adb -s "$SERIAL" pull "$PHONE_DIR/run.log" .
```

The last argument is the number of complete 80 ms frames. Start with `1` or
`2`; `38` processes the tested 3.04 s audio. The input must be 24 kHz mono
PCM16 WAV. The code file has one CSV row per frame: frame index followed by
eight codebook tokens. `encoder_ms` and `decoder_ms` in the log time only
`graphExecute`, excluding model load, audio I/O, and playback.

## Validation boundary

On the tested 3.04 s recording, ONNX Encoder codes matched PyTorch streaming
for all 304 tokens. The phone QNN Encoder matched 298/304; six tokens differed
on frames 7, 18, 25, and 37. For the *same QNN codes*, phone QNN Decoder
audio versus ONNX Decoder had RMSE `0.000290` and relative RMS error `0.00533`.
The complete phone loopback versus PyTorch had RMSE `0.00480`, relative RMS
error `0.0883`, and peak absolute error `0.150`; the larger difference follows
the discrete Encoder token changes. Thus this is a working float codec
demonstration and a candidate for further quality review, not exact
PyTorch audio parity. Listen to both WAV files before a subjective claim.

The host processes an existing WAV in sequence; it is not a microphone or
speaker app. For a live demo, keep the same per-component state loop and add
audio capture, buffering, playback, stream reset, and end-to-end latency
measurement around it. `scripts/moshi_compare_mimi_phone.py` compares an
output WAV and code CSV to PyTorch streaming and ONNX references. It requires
the real checkpoint and the explicit-state ONNX bundle on the Linux host.
