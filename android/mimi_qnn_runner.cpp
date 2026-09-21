// Minimal on-device Mimi DLC loopback: PCM16 WAV -> QNN encoder -> QNN decoder -> WAV.
// Uses the QAIRT 2.45 public QNN interfaces. Each graph owns its streaming state.

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

namespace {

constexpr size_t kFrameSamples = 1920;
constexpr uint32_t kSampleRate = 24000;

void check(Qnn_ErrorHandle_t status, const std::string& operation) {
  if (status != QNN_SUCCESS) {
    throw std::runtime_error(operation + " failed, QNN status " +
                             std::to_string(static_cast<unsigned long long>(status)));
  }
}

void* openLibrary(const std::string& path) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) throw std::runtime_error("dlopen " + path + ": " + dlerror());
  return handle;
}

template <typename Function>
Function symbol(void* handle, const char* name) {
  void* result = dlsym(handle, name);
  if (!result) throw std::runtime_error(std::string("dlsym ") + name + ": " + dlerror());
  return reinterpret_cast<Function>(result);
}

const char* tensorName(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) return tensor.v1.name;
  if (tensor.version == QNN_TENSOR_VERSION_2) return tensor.v2.name;
  throw std::runtime_error("Unsupported QNN tensor descriptor version");
}

Qnn_DataType_t tensorType(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) return tensor.v1.dataType;
  if (tensor.version == QNN_TENSOR_VERSION_2) return tensor.v2.dataType;
  throw std::runtime_error("Unsupported QNN tensor descriptor version");
}

uint32_t tensorRank(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) return tensor.v1.rank;
  if (tensor.version == QNN_TENSOR_VERSION_2) return tensor.v2.rank;
  throw std::runtime_error("Unsupported QNN tensor descriptor version");
}

const uint32_t* tensorDimensions(const Qnn_Tensor_t& tensor) {
  if (tensor.version == QNN_TENSOR_VERSION_1) return tensor.v1.dimensions;
  if (tensor.version == QNN_TENSOR_VERSION_2) return tensor.v2.dimensions;
  throw std::runtime_error("Unsupported QNN tensor descriptor version");
}

void setClientBuffer(Qnn_Tensor_t& tensor, void* data, uint32_t length) {
  if (tensor.version == QNN_TENSOR_VERSION_1) {
    tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    tensor.v1.clientBuf = {data, length};
  } else if (tensor.version == QNN_TENSOR_VERSION_2) {
    tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
    tensor.v2.clientBuf = {data, length};
  } else {
    throw std::runtime_error("Unsupported QNN tensor descriptor version");
  }
}

size_t tensorBytes(const Qnn_Tensor_t& tensor) {
  size_t width;
  switch (tensorType(tensor)) {
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
      width = 4;
      break;
    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64:
      width = 8;
      break;
    default:
      throw std::runtime_error(std::string("Unsupported tensor datatype for ") +
                               tensorName(tensor) + ": " +
                               std::to_string(tensorType(tensor)));
  }
  size_t elements = 1;
  for (uint32_t i = 0; i < tensorRank(tensor); ++i) {
    const uint32_t dimension = tensorDimensions(tensor)[i];
    if (dimension == 0 || elements > SIZE_MAX / dimension) {
      throw std::runtime_error(std::string("Invalid tensor dimensions for ") +
                               tensorName(tensor));
    }
    elements *= dimension;
  }
  if (elements > SIZE_MAX / width) throw std::runtime_error("Tensor byte size overflow");
  return elements * width;
}

struct Buffer {
  Qnn_Tensor_t tensor{};
  std::vector<uint8_t> bytes;
  std::string name;
};

class Runtime {
 public:
  explicit Runtime(const std::string& libraryDir) {
    backendLibrary_ = openLibrary(libraryDir + "/libQnnHtp.so");
    systemLibrary_ = openLibrary(libraryDir + "/libQnnSystem.so");

    auto getQnn = symbol<decltype(&QnnInterface_getProviders)>(
        backendLibrary_, "QnnInterface_getProviders");
    const QnnInterface_t** providers = nullptr;
    uint32_t count = 0;
    check(getQnn(&providers, &count), "QnnInterface_getProviders");
    for (uint32_t i = 0; i < count; ++i) {
      const auto& version = providers[i]->apiVersion.coreApiVersion;
      if (version.major == QNN_API_VERSION_MAJOR &&
          version.minor >= QNN_API_VERSION_MINOR) {
        qnnProvider_ = providers[i];
        break;
      }
    }
    if (!qnnProvider_) throw std::runtime_error("No compatible QNN interface");
    qnn = qnnProvider_->QNN_INTERFACE_VER_NAME;

    auto getSystem = symbol<decltype(&QnnSystemInterface_getProviders)>(
        systemLibrary_, "QnnSystemInterface_getProviders");
    const QnnSystemInterface_t** systemProviders = nullptr;
    count = 0;
    check(getSystem(&systemProviders, &count), "QnnSystemInterface_getProviders");
    for (uint32_t i = 0; i < count; ++i) {
      const auto& version = systemProviders[i]->systemApiVersion;
      if (version.major == QNN_SYSTEM_API_VERSION_MAJOR &&
          version.minor >= QNN_SYSTEM_API_VERSION_MINOR) {
        system = systemProviders[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
        break;
      }
    }
    if (!system.systemDlcCreateFromFile || !system.systemDlcComposeGraphs) {
      throw std::runtime_error("No compatible QNN System DLC interface");
    }
    check(qnn.backendCreate(nullptr, nullptr, &backend), "backendCreate");
    const auto deviceStatus = qnn.deviceCreate(nullptr, nullptr, &device);
    if (deviceStatus != QNN_SUCCESS &&
        deviceStatus != QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE) {
      check(deviceStatus, "deviceCreate");
    }
    std::cerr << "QNN HTP and System DLC interfaces ready\n";
  }

  ~Runtime() {
    if (device && qnn.deviceFree) qnn.deviceFree(device);
    if (backend && qnn.backendFree) qnn.backendFree(backend);
    // The HTP backend can retain process-wide cleanup hooks, so leave its SO loaded.
  }

  const QnnInterface_t& provider() const { return *qnnProvider_; }
  QNN_INTERFACE_VER_TYPE qnn{};
  QNN_SYSTEM_INTERFACE_VER_TYPE system{};
  Qnn_BackendHandle_t backend = nullptr;
  Qnn_DeviceHandle_t device = nullptr;

 private:
  void* backendLibrary_ = nullptr;
  void* systemLibrary_ = nullptr;
  const QnnInterface_t* qnnProvider_ = nullptr;
};

class Graph {
 public:
  Graph(Runtime& runtime, const std::string& dlcPath) : runtime_(runtime) {
    check(runtime_.qnn.contextCreate(runtime_.backend, runtime_.device, nullptr, &context_),
          "contextCreate " + dlcPath);
    check(runtime_.system.systemDlcCreateFromFile(nullptr, dlcPath.c_str(), &dlc_),
          "systemDlcCreateFromFile " + dlcPath);
    uint32_t count = 0;
    check(runtime_.system.systemDlcComposeGraphs(
              dlc_, nullptr, 0, runtime_.backend, context_, runtime_.provider(),
              QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1, &graphs_, &count),
          "systemDlcComposeGraphs " + dlcPath);
    if (count != 1 || !graphs_ || graphs_[0].version !=
                                  QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
      throw std::runtime_error("Expected one DLC graph with v1 graph metadata");
    }
    const auto& graphInfo = graphs_[0].graphInfoV1;
    name_ = graphInfo.graphName;
    check(runtime_.qnn.graphRetrieve(context_, name_.c_str(), &graph_),
          "graphRetrieve " + name_);
    check(runtime_.qnn.graphFinalize(graph_, nullptr, nullptr), "graphFinalize " + name_);
    prepare(graphInfo.graphInputs, graphInfo.numGraphInputs, inputs_, inputTensors_);
    prepare(graphInfo.graphOutputs, graphInfo.numGraphOutputs, outputs_, outputTensors_);
    std::cerr << dlcPath << " graph=" << name_ << '\n';
    printIO("input", inputs_);
    printIO("output", outputs_);
  }

  ~Graph() {
    if (context_ && runtime_.qnn.contextFree) runtime_.qnn.contextFree(context_, nullptr);
    if (graphs_) std::free(graphs_);
    if (dlc_ && runtime_.system.systemDlcFree) runtime_.system.systemDlcFree(dlc_);
  }

  Buffer& input(const std::string& name) { return find(inputs_, name); }
  Buffer& output(const std::string& name) {
    for (auto& buffer : outputs_) {
      if (buffer.name == name) return buffer;
    }
    // AI Hub DLCs preserve input names, but export outputs as output_0..3.
    if (name == "codes" || name == "audio") return find(outputs_, "output_0");
    if (name == "position_out") return find(outputs_, "output_1");
    if (name == "kv_cache_out") return find(outputs_, "output_2");
    if (name == "conv_state_out") return find(outputs_, "output_3");
    throw std::runtime_error("Missing QNN graph output: " + name);
  }

  double execute() {
    const auto start = std::chrono::steady_clock::now();
    check(runtime_.qnn.graphExecute(graph_, inputTensors_.data(), inputTensors_.size(),
                                    outputTensors_.data(), outputTensors_.size(), nullptr,
                                    nullptr),
          "graphExecute " + name_);
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  void advanceState() {
    for (const auto& name : {"position", "kv_cache", "conv_state"}) {
      auto& source = output(std::string(name) + "_out");
      auto& target = input(name);
      copy(source, target);
    }
  }

  static void copy(const Buffer& source, Buffer& target) {
    if (tensorType(source.tensor) != tensorType(target.tensor) ||
        source.bytes.size() != target.bytes.size()) {
      throw std::runtime_error("Incompatible tensor copy: " + source.name + " -> " + target.name);
    }
    std::memcpy(target.bytes.data(), source.bytes.data(), source.bytes.size());
  }

 private:
  static Buffer& find(std::vector<Buffer>& buffers, const std::string& name) {
    for (auto& buffer : buffers) {
      if (buffer.name == name) return buffer;
    }
    throw std::runtime_error("Missing QNN graph tensor: " + name);
  }

  static void prepare(Qnn_Tensor_t* metadata, uint32_t count,
                      std::vector<Buffer>& buffers, std::vector<Qnn_Tensor_t>& tensors) {
    if (count != 4) throw std::runtime_error("Expected four Mimi graph tensors per direction");
    buffers.resize(count);
    tensors.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
      auto& buffer = buffers[i];
      buffer.tensor = metadata[i];
      buffer.name = tensorName(metadata[i]);
      buffer.bytes.resize(tensorBytes(metadata[i]));
      setClientBuffer(buffer.tensor, buffer.bytes.data(),
                      static_cast<uint32_t>(buffer.bytes.size()));
      tensors[i] = buffer.tensor;
    }
  }

  static void printIO(const char* kind, const std::vector<Buffer>& buffers) {
    for (const auto& buffer : buffers) {
      std::cerr << "  " << kind << ' ' << buffer.name << " dtype=" <<
          tensorType(buffer.tensor) << " version=" << buffer.tensor.version
                << " shape=[";
      for (uint32_t i = 0; i < tensorRank(buffer.tensor); ++i) {
        if (i) std::cerr << ',';
        std::cerr << tensorDimensions(buffer.tensor)[i];
      }
      std::cerr << "] bytes=" << buffer.bytes.size() << '\n';
    }
  }

  Runtime& runtime_;
  Qnn_ContextHandle_t context_ = nullptr;
  QnnSystemDlc_Handle_t dlc_ = nullptr;
  QnnSystemContext_GraphInfo_t* graphs_ = nullptr;
  Qnn_GraphHandle_t graph_ = nullptr;
  std::string name_;
  std::vector<Buffer> inputs_;
  std::vector<Buffer> outputs_;
  std::vector<Qnn_Tensor_t> inputTensors_;
  std::vector<Qnn_Tensor_t> outputTensors_;
};

uint16_t read16(const uint8_t* data) {
  return uint16_t(data[0]) | uint16_t(data[1]) << 8;
}

uint32_t read32(const uint8_t* data) {
  return uint32_t(data[0]) | uint32_t(data[1]) << 8 |
         uint32_t(data[2]) << 16 | uint32_t(data[3]) << 24;
}

void write16(std::ostream& output, uint16_t value) {
  output.put(char(value & 255));
  output.put(char((value >> 8) & 255));
}

void write32(std::ostream& output, uint32_t value) {
  write16(output, uint16_t(value & 65535));
  write16(output, uint16_t(value >> 16));
}

std::vector<int16_t> readWav(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open input WAV: " + path);
  std::vector<uint8_t> file((std::istreambuf_iterator<char>(input)), {});
  if (file.size() < 44 || std::memcmp(file.data(), "RIFF", 4) ||
      std::memcmp(file.data() + 8, "WAVE", 4)) {
    throw std::runtime_error("Expected RIFF/WAVE input");
  }
  bool formatSeen = false;
  std::vector<int16_t> samples;
  for (size_t cursor = 12; cursor + 8 <= file.size();) {
    const uint32_t length = read32(file.data() + cursor + 4);
    const size_t start = cursor + 8;
    if (length > file.size() - start) throw std::runtime_error("Malformed WAV chunk");
    if (!std::memcmp(file.data() + cursor, "fmt ", 4)) {
      if (length < 16 || read16(file.data() + start) != 1 ||
          read16(file.data() + start + 2) != 1 ||
          read32(file.data() + start + 4) != kSampleRate ||
          read16(file.data() + start + 14) != 16) {
        throw std::runtime_error("Expected 24 kHz mono PCM16 WAV");
      }
      formatSeen = true;
    } else if (!std::memcmp(file.data() + cursor, "data", 4)) {
      if (length % 2) throw std::runtime_error("Odd PCM16 data size");
      samples.resize(length / 2);
      for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<int16_t>(read16(file.data() + start + 2 * i));
      }
    }
    cursor = start + length + (length & 1);
  }
  if (!formatSeen || samples.empty()) throw std::runtime_error("Missing WAV fmt or data chunk");
  return samples;
}

void writeWav(const std::string& path, const std::vector<int16_t>& samples) {
  const uint32_t dataBytes = uint32_t(samples.size() * 2);
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("Cannot create output WAV: " + path);
  output.write("RIFF", 4);
  write32(output, 36 + dataBytes);
  output.write("WAVEfmt ", 8);
  write32(output, 16);
  write16(output, 1);
  write16(output, 1);
  write32(output, kSampleRate);
  write32(output, kSampleRate * 2);
  write16(output, 2);
  write16(output, 16);
  output.write("data", 4);
  write32(output, dataBytes);
  for (int16_t sample : samples) write16(output, static_cast<uint16_t>(sample));
  if (!output) throw std::runtime_error("Failed writing output WAV");
}

void expect(Buffer& buffer, Qnn_DataType_t type, size_t bytes) {
  if (tensorType(buffer.tensor) != type || buffer.bytes.size() != bytes) {
    throw std::runtime_error("Unexpected tensor contract: " + buffer.name);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      std::cerr << "Usage: mimi_qnn_runner LIB_DIR ENCODER.dlc DECODER.dlc "
                   "INPUT.wav OUTPUT.wav FRAMES\n";
      return 2;
    }
    const size_t requestedFrames = std::stoul(argv[6]);
    const auto samples = readWav(argv[4]);
    if (requestedFrames == 0 || requestedFrames > samples.size() / kFrameSamples) {
      throw std::runtime_error("Requested frames exceed available complete 80 ms WAV frames");
    }
    Runtime runtime(argv[1]);
    Graph encoder(runtime, argv[2]);
    Graph decoder(runtime, argv[3]);
    auto& audioInput = encoder.input("audio");
    auto& codesOutput = encoder.output("codes");
    auto& codesInput = decoder.input("codes");
    auto& audioOutput = decoder.output("audio");
    expect(audioInput, QNN_DATATYPE_FLOAT_32, kFrameSamples * sizeof(float));
    expect(codesOutput, QNN_DATATYPE_INT_32, 8 * sizeof(int32_t));
    expect(codesInput, QNN_DATATYPE_INT_32, 8 * sizeof(int32_t));
    expect(audioOutput, QNN_DATATYPE_FLOAT_32, kFrameSamples * sizeof(float));
    std::vector<int16_t> reconstructed;
    reconstructed.reserve(requestedFrames * kFrameSamples);
    std::ofstream codesFile(std::string(argv[5]) + ".codes.txt");
    if (!codesFile) throw std::runtime_error("Cannot create codes file");
    for (size_t frame = 0; frame < requestedFrames; ++frame) {
      std::array<float, kFrameSamples> pcm{};
      for (size_t i = 0; i < kFrameSamples; ++i) {
        pcm[i] = float(samples[frame * kFrameSamples + i]) / 32768.0f;
      }
      std::memcpy(audioInput.bytes.data(), pcm.data(), audioInput.bytes.size());
      const double encoderMs = encoder.execute();
      Graph::copy(codesOutput, codesInput);
      int32_t encoderPosition;
      std::memcpy(&encoderPosition, encoder.output("position_out").bytes.data(),
                  sizeof(encoderPosition));
      encoder.advanceState();
      const double decoderMs = decoder.execute();
      int32_t decoderPosition;
      std::memcpy(&decoderPosition, decoder.output("position_out").bytes.data(),
                  sizeof(decoderPosition));
      decoder.advanceState();
      const int32_t expectedPosition = static_cast<int32_t>(2 * (frame + 1));
      if (encoderPosition != expectedPosition || decoderPosition != expectedPosition) {
        throw std::runtime_error("Mimi streaming position did not advance by two");
      }
      for (size_t i = 0; i < kFrameSamples; ++i) {
        float value;
        std::memcpy(&value, audioOutput.bytes.data() + i * sizeof(float), sizeof(float));
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite decoder audio");
        reconstructed.push_back(static_cast<int16_t>(std::lrintf(
            std::clamp(value, -1.0f, 32767.0f / 32768.0f) * 32768.0f)));
      }
      codesFile << frame;
      for (size_t i = 0; i < 8; ++i) {
        int32_t code;
        std::memcpy(&code, codesOutput.bytes.data() + i * sizeof(int32_t), sizeof(code));
        codesFile << ',' << code;
      }
      codesFile << '\n';
      std::cerr << "frame=" << frame << " encoder_ms=" << encoderMs
                << " decoder_ms=" << decoderMs << '\n';
    }
    if (!codesFile) throw std::runtime_error("Failed writing codes file");
    writeWav(argv[5], reconstructed);
    std::cerr << "Wrote " << reconstructed.size() << " samples to " << argv[5] << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Mimi QNN runner error: " << error.what() << '\n';
    return 1;
  }
}
