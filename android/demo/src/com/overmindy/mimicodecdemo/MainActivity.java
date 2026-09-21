package com.overmindy.mimicodecdemo;

import android.Manifest;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity {
    private static final int PICK_AUDIO = 1;
    private static final int PICK_ENCODER = 2;
    private static final int PICK_DECODER = 3;
    private static final int REQUEST_MICROPHONE = 4;
    private static final int SAMPLE_RATE = 24_000;
    private static final int FRAME_SAMPLES = 1920;
    private static final int BACKGROUND = Color.rgb(10, 17, 29);
    private static final int CARD = Color.rgb(26, 39, 57);
    private static final int ACCENT = Color.rgb(68, 222, 194);
    private final File[] files = new File[4];
    private TextView status;
    private TextView modelStatus;
    private TextView audioStatus;
    private TextView tokens;
    private Button loadButton;
    private Button recordButton;
    private Button runButton;
    private Button originalButton;
    private Button outputButton;
    private Button exportButton;
    private MediaPlayer player;
    private boolean working;
    private volatile boolean recording;
    private volatile long sessionHandle;
    private final ExecutorService worker = Executors.newSingleThreadExecutor();

    private static native long nativePrepare(String libraryDir, String encoderDlc,
                                             String decoderDlc);
    private static native int nativeRunPrepared(long handle, String inputWav,
                                                 String outputWav);
    private static native void nativeRelease(long handle);

    static {
        System.loadLibrary("mimi_codec");
    }

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().setStatusBarColor(BACKGROUND);
        getWindow().setNavigationBarColor(BACKGROUND);
        files[PICK_AUDIO] = new File(getFilesDir(), "input.wav");
        files[PICK_ENCODER] = new File(getFilesDir(), "mimi_encoder.dlc");
        files[PICK_DECODER] = new File(getFilesDir(), "mimi_decoder.dlc");

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(BACKGROUND);
        LinearLayout column = new LinearLayout(this);
        column.setOrientation(LinearLayout.VERTICAL);
        column.setPadding(dp(20), dp(26), dp(20), dp(26));
        scroll.addView(column);
        setContentView(scroll);

        label(column, "MIMI  •  QNN FLOAT CODEC", 13, ACCENT, true);
        label(column, "听见编码的过程", 28, Color.WHITE, true);
        label(column, "WAV → Encoder → 8 个 codec tokens → Decoder → WAV",
              15, Color.rgb(176, 193, 210), false);

        LinearLayout steps = card(column);
        label(steps, "① 导入并加载模型", 19, Color.WHITE, true);
        label(steps, "两个 DLC 只需导入一次。点击加载，等 QNN 图就绪后再录音或推理。",
              14, Color.rgb(176, 193, 210), false);
        button(steps, "导入 Encoder DLC", () -> pick(PICK_ENCODER));
        button(steps, "导入 Decoder DLC", () -> pick(PICK_DECODER));
        loadButton = button(steps, "加载两个 DLC 到 QNN", this::loadModels);
        modelStatus = label(steps, "等待导入两个 DLC", 14,
                            Color.rgb(176, 193, 210), false);

        LinearLayout input = card(column);
        label(input, "② 准备音频", 19, Color.WHITE, true);
        label(input, "录音直接保存为 24 kHz 单声道 PCM16 WAV；点击停止后再推理。也可选已有 WAV。",
              14, Color.rgb(176, 193, 210), false);
        recordButton = button(input, "● 开始录音", this::toggleRecording);
        button(input, "选择 24 kHz 单声道 WAV", () -> pick(PICK_AUDIO));
        button(input, "使用内置演示音频", this::useDemoAudio);
        audioStatus = label(input, "等待模型就绪", 14,
                            Color.rgb(176, 193, 210), false);

        LinearLayout result = card(column);
        label(result, "③ 手机本地编码与重建", 19, Color.WHITE, true);
        runButton = button(result, "运行 QNN Encoder → Decoder", this::runCodec);
        status = label(result, "加载模型并准备音频后可推理", 14,
                       Color.rgb(176, 193, 210), false);
        originalButton = button(result, "▶ 播放原音", () -> play(files[PICK_AUDIO]));
        outputButton = button(result, "▶ 播放重建音频", () -> play(outputFile()));
        exportButton = button(result, "保存 WAV 与 tokens 到下载目录", this::exportResults);

        LinearLayout tokenCard = card(column);
        label(tokenCard, "④ 每帧的 codec tokens", 19, Color.WHITE, true);
        label(tokenCard, "每行是 80 ms 音频对应的 8 个 codebook ID。它们直接来自手机上的 QNN Encoder。",
              14, Color.rgb(176, 193, 210), false);
        tokens = label(tokenCard, "运行后显示编码结果", 13, ACCENT, false);
        tokens.setTypeface(Typeface.MONOSPACE);
        if (files[PICK_ENCODER].isFile() && files[PICK_DECODER].isFile()) {
            modelStatus.setText("DLC 已导入；点击加载两个 DLC 到 QNN");
        }
        if (files[PICK_AUDIO].isFile()) audioStatus.setText("已有输入音频，可直接推理");
        refresh();
        scroll.setFocusableInTouchMode(true);
        scroll.requestFocus();
        scroll.post(() -> scroll.scrollTo(0, 0));
    }

    private File outputFile() { return new File(getFilesDir(), "reconstructed.wav"); }
    private File codesFile() { return new File(getFilesDir(), "reconstructed.wav.codes.txt"); }

    private void pick(int kind) {
        if (working || recording) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(intent, kind);
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (result != RESULT_OK || data == null || data.getData() == null ||
            request < PICK_AUDIO || request > PICK_DECODER) return;
        Uri uri = data.getData();
        doCopy(() -> getContentResolver().openInputStream(uri), files[request],
               request == PICK_AUDIO ? "音频已导入" : "DLC 已导入");
    }

    private interface Source { InputStream open() throws IOException; }

    private void useDemoAudio() {
        doCopy(() -> getAssets().open("demo.wav"), files[PICK_AUDIO],
               "演示音频已导入");
    }

    private void doCopy(Source source, File target, String success) {
        if (working || recording) return;
        stopPlayer();
        outputFile().delete();
        codesFile().delete();
        tokens.setText("输入或模型已更换；请重新运行编码");
        boolean modelChanged = target == files[PICK_ENCODER] ||
                               target == files[PICK_DECODER];
        long oldHandle = modelChanged ? sessionHandle : 0;
        if (modelChanged) {
            sessionHandle = 0;
            modelStatus.setText("模型文件已更换，请重新加载 DLC");
        }
        working = true;
        refresh();
        status.setText("正在导入 " + target.getName() + "…");
        worker.execute(() -> {
            if (oldHandle != 0) nativeRelease(oldHandle);
            try (InputStream input = source.open();
                 OutputStream output = new FileOutputStream(target)) {
                byte[] buffer = new byte[1024 * 1024];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
                if (target.length() == 0) throw new IOException("文件为空");
                runOnUiThread(() -> {
                    if (modelChanged) modelStatus.setText("DLC 已导入；点击加载两个 DLC 到 QNN");
                    else audioStatus.setText(success + "：" + target.getName());
                    status.setText(success + "：" + target.getName());
                });
            } catch (Exception error) {
                target.delete();
                runOnUiThread(() -> status.setText("导入失败：" + error.getMessage()));
            } finally {
                runOnUiThread(() -> { working = false; refresh(); });
            }
        });
    }

    private void loadModels() {
        if (working || recording || !modelFilesReady()) return;
        working = true;
        modelStatus.setText("正在加载并初始化两个 QNN 图…");
        refresh();
        worker.execute(() -> {
            long start = System.nanoTime();
            try {
                long previous = sessionHandle;
                sessionHandle = 0;
                if (previous != 0) nativeRelease(previous);
                sessionHandle = nativePrepare(getApplicationInfo().nativeLibraryDir,
                        files[PICK_ENCODER].getAbsolutePath(),
                        files[PICK_DECODER].getAbsolutePath());
                double seconds = (System.nanoTime() - start) / 1e9;
                runOnUiThread(() -> modelStatus.setText(String.format(Locale.US,
                        "两个 DLC 已就绪；本次加载用时 %.1f 秒", seconds)));
            } catch (Exception error) {
                runOnUiThread(() -> modelStatus.setText("加载失败：" + error.getMessage()));
            } finally {
                runOnUiThread(() -> { working = false; refresh(); });
            }
        });
    }

    private void toggleRecording() {
        if (recording) {
            recording = false;
            recordButton.setText("正在结束录音…");
            audioStatus.setText("正在保存完整的 80 ms 音频帧…");
            return;
        }
        if (working || sessionHandle == 0) return;
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) !=
                PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO},
                               REQUEST_MICROPHONE);
            return;
        }
        beginRecording();
    }

    @Override
    public void onRequestPermissionsResult(int request, String[] permissions,
                                            int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (request != REQUEST_MICROPHONE) return;
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
            beginRecording();
        } else {
            audioStatus.setText("需要麦克风权限才能录音；也可选择现有 WAV");
        }
    }

    private void beginRecording() {
        if (working || recording || sessionHandle == 0) return;
        stopPlayer();
        outputFile().delete();
        codesFile().delete();
        tokens.setText("录音后点击推理查看 codec tokens");
        recording = true;
        audioStatus.setText("正在录音；点击停止后生成 24 kHz PCM16 WAV");
        refresh();
        worker.execute(() -> {
            File temporary = new File(getFilesDir(), "recording.tmp.wav");
            AudioRecord microphone = null;
            try {
                int minimumBytes = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
                if (minimumBytes <= 0) throw new IOException("本机不支持 24 kHz 单声道录音");
                microphone = new AudioRecord(MediaRecorder.AudioSource.MIC,
                        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        Math.max(minimumBytes, FRAME_SAMPLES * 4));
                if (microphone.getState() != AudioRecord.STATE_INITIALIZED) {
                    throw new IOException("麦克风初始化失败");
                }
                short[] pcm = new short[FRAME_SAMPLES];
                byte[] packed = new byte[FRAME_SAMPLES * 2];
                long sampleCount = 0;
                try (RandomAccessFile wav = new RandomAccessFile(temporary, "rw")) {
                    wav.setLength(0);
                    wav.write(new byte[44]);
                    microphone.startRecording();
                    while (recording) {
                        int count = microphone.read(pcm, 0, pcm.length,
                                AudioRecord.READ_BLOCKING);
                        if (count < 0) throw new IOException("录音读取失败：" + count);
                        for (int i = 0; i < count; i++) {
                            packed[2 * i] = (byte) pcm[i];
                            packed[2 * i + 1] = (byte) (pcm[i] >>> 8);
                        }
                        wav.write(packed, 0, count * 2);
                        sampleCount += count;
                        if (sampleCount > (0xffffffffL - 36) / 2) {
                            throw new IOException("录音超过 WAV 格式容量");
                        }
                    }
                    long completeSamples = sampleCount - sampleCount % FRAME_SAMPLES;
                    if (completeSamples == 0) throw new IOException("请至少录制 80 ms");
                    wav.setLength(44 + completeSamples * 2);
                    wav.seek(0);
                    writeWavHeader(wav, completeSamples);
                    sampleCount = completeSamples;
                }
                files[PICK_AUDIO].delete();
                if (!temporary.renameTo(files[PICK_AUDIO])) {
                    throw new IOException("录音文件保存失败");
                }
                final long savedSamples = sampleCount;
                runOnUiThread(() -> audioStatus.setText(String.format(Locale.US,
                        "录音完成：%.2f 秒 / %d 帧；点击推理",
                        savedSamples / (double) SAMPLE_RATE,
                        savedSamples / FRAME_SAMPLES)));
            } catch (Exception error) {
                temporary.delete();
                runOnUiThread(() -> audioStatus.setText("录音失败：" + error.getMessage()));
            } finally {
                if (microphone != null) {
                    try { microphone.stop(); } catch (IllegalStateException ignored) { }
                    microphone.release();
                }
                recording = false;
                runOnUiThread(this::refresh);
            }
        });
    }

    private static void writeLittle16(RandomAccessFile wav, int value) throws IOException {
        wav.write(value & 255);
        wav.write((value >>> 8) & 255);
    }

    private static void writeLittle32(RandomAccessFile wav, long value) throws IOException {
        writeLittle16(wav, (int) value);
        writeLittle16(wav, (int) (value >>> 16));
    }

    private static void writeWavHeader(RandomAccessFile wav, long samples)
            throws IOException {
        long bytes = samples * 2;
        wav.write("RIFF".getBytes(StandardCharsets.US_ASCII));
        writeLittle32(wav, 36 + bytes);
        wav.write("WAVEfmt ".getBytes(StandardCharsets.US_ASCII));
        writeLittle32(wav, 16);
        writeLittle16(wav, 1);
        writeLittle16(wav, 1);
        writeLittle32(wav, SAMPLE_RATE);
        writeLittle32(wav, SAMPLE_RATE * 2);
        writeLittle16(wav, 2);
        writeLittle16(wav, 16);
        wav.write("data".getBytes(StandardCharsets.US_ASCII));
        writeLittle32(wav, bytes);
    }

    private void runCodec() {
        if (working || recording || sessionHandle == 0 || !files[PICK_AUDIO].isFile()) return;
        stopPlayer();
        outputFile().delete();
        codesFile().delete();
        tokens.setText("正在执行 QNN 图…");
        status.setText("模型已加载；正在逐帧执行 Encoder → Decoder…");
        working = true;
        refresh();
        worker.execute(() -> {
            long start = System.nanoTime();
            try {
                int frames = nativeRunPrepared(sessionHandle,
                                       files[PICK_AUDIO].getAbsolutePath(),
                                       outputFile().getAbsolutePath());
                String codeText = readCodes();
                double seconds = (System.nanoTime() - start) / 1e9;
                runOnUiThread(() -> {
                    status.setText(String.format(Locale.US,
                            "推理完成：%d 帧 / %.2f 秒音频；已加载模型推理用时 %.2f 秒（音频时长的 %.2f 倍）",
                            frames, frames * 0.08, seconds,
                            frames * 0.08 / Math.max(seconds, 1e-9)));
                    tokens.setText(codeText);
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    status.setText("运行失败：" + error.getMessage());
                    tokens.setText("没有生成有效 tokens");
                });
            } finally {
                runOnUiThread(() -> { working = false; refresh(); });
            }
        });
    }

    private String readCodes() throws IOException {
        StringBuilder text = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(codesFile()), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                text.append(String.format(java.util.Locale.US, "%02d  ",
                                          Integer.parseInt(values[0])));
                for (int i = 1; i < values.length; i++) {
                    if (i > 1) text.append(' ');
                    text.append(values[i]);
                }
                text.append('\n');
            }
        }
        return text.toString();
    }

    private void play(File file) {
        if (!file.isFile()) return;
        stopPlayer();
        try {
            player = new MediaPlayer();
            player.setDataSource(file.getAbsolutePath());
            player.setOnCompletionListener(done -> stopPlayer());
            player.prepare();
            player.start();
            status.setText("正在播放：" + file.getName());
        } catch (IOException error) {
            stopPlayer();
            status.setText("播放失败：" + error.getMessage());
        }
    }

    private void stopPlayer() {
        if (player != null) { player.release(); player = null; }
    }

    @Override
    protected void onDestroy() {
        recording = false;
        stopPlayer();
        worker.execute(() -> {
            long handle = sessionHandle;
            sessionHandle = 0;
            if (handle != 0) nativeRelease(handle);
        });
        worker.shutdown();
        super.onDestroy();
    }

    private void exportResults() {
        if (!outputFile().isFile() || !codesFile().isFile()) return;
        working = true;
        refresh();
        status.setText("正在保存到 Downloads/MimiDemo…");
        worker.execute(() -> {
            try {
                String suffix = Long.toString(System.currentTimeMillis());
                saveDownload(outputFile(), "mimi-" + suffix + ".wav", "audio/wav");
                saveDownload(codesFile(), "mimi-" + suffix + "-tokens.csv", "text/csv");
                runOnUiThread(() -> status.setText("已保存到 Downloads/MimiDemo"));
            } catch (Exception error) {
                runOnUiThread(() -> status.setText("保存失败：" + error.getMessage()));
            } finally {
                runOnUiThread(() -> { working = false; refresh(); });
            }
        });
    }

    private void saveDownload(File source, String name, String mime) throws IOException {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Downloads.DISPLAY_NAME, name);
        values.put(MediaStore.Downloads.MIME_TYPE, mime);
        values.put(MediaStore.Downloads.RELATIVE_PATH,
                   Environment.DIRECTORY_DOWNLOADS + "/MimiDemo");
        Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI,
                                              values);
        if (uri == null) throw new IOException("无法创建下载文件");
        try (InputStream input = new FileInputStream(source);
             OutputStream output = getContentResolver().openOutputStream(uri)) {
            if (output == null) throw new IOException("无法写入下载文件");
            byte[] buffer = new byte[65536];
            int count;
            while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
        }
    }

    private boolean modelFilesReady() {
        return files[PICK_ENCODER].isFile() && files[PICK_DECODER].isFile();
    }

    private void refresh() {
        loadButton.setEnabled(!working && !recording && modelFilesReady());
        recordButton.setEnabled(!working && (recording || sessionHandle != 0));
        recordButton.setText(recording ? "■ 停止录音" : "● 开始录音");
        runButton.setEnabled(!working && !recording && sessionHandle != 0 &&
                             files[PICK_AUDIO].isFile());
        originalButton.setEnabled(!working && !recording && files[PICK_AUDIO].isFile());
        outputButton.setEnabled(!working && !recording && outputFile().isFile());
        exportButton.setEnabled(!working && !recording && outputFile().isFile() &&
                                codesFile().isFile());
    }

    private LinearLayout card(LinearLayout parent) {
        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setPadding(dp(16), dp(16), dp(16), dp(16));
        android.graphics.drawable.GradientDrawable background =
                new android.graphics.drawable.GradientDrawable();
        background.setColor(CARD);
        background.setCornerRadius(dp(18));
        panel.setBackground(background);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(20);
        parent.addView(panel, params);
        return panel;
    }

    private TextView label(LinearLayout parent, String value, int size,
                           int color, boolean bold) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextColor(color);
        view.setTextSize(size);
        view.setPadding(0, dp(4), 0, dp(8));
        if (bold) view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        parent.addView(view);
        return view;
    }

    private Button button(LinearLayout parent, String title, Runnable action) {
        Button button = new Button(this);
        button.setText(title);
        button.setAllCaps(false);
        button.setTextColor(BACKGROUND);
        button.setBackgroundTintList(android.content.res.ColorStateList.valueOf(ACCENT));
        button.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(8);
        parent.addView(button, params);
        button.setOnClickListener(view -> action.run());
        return button;
    }

    private int dp(int value) {
        return Math.round(getResources().getDisplayMetrics().density * value);
    }
}
