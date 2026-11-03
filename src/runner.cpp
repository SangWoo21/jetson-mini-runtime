#include "minirt/runner.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace minirt {

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess)                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                 \
                               cudaGetErrorString(e));                     \
  } while (0)

void Logger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
}

static size_t volume(const nvinfer1::Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    if (d.d[i] < 0) throw std::runtime_error("dynamic shape not supported (week 1)");
    v *= static_cast<size_t>(d.d[i]);
  }
  return v;
}

Runner::Runner(const std::string& engine_path) {
  // 1) 엔진 파일 읽기
  std::ifstream f(engine_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + engine_path);
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});

  // 2) 역직렬화 (TRT 8.x는 delete로 해제 가능 -> unique_ptr 사용)
  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) throw std::runtime_error("createInferRuntime failed");
  engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
  if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
  context_.reset(engine_->createExecutionContext());
  if (!context_) throw std::runtime_error("createExecutionContext failed");

  // 3) I/O 텐서를 '이름'으로 찾는다 (TRT 8.5 API. 바인딩 인덱스 방식은 deprecated)
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
      throw std::runtime_error(std::string("non-FP32 I/O: ") + name);
    size_t n = volume(engine_->getTensorShape(name));
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      in_name_ = name; in_count_ = n;
    } else {
      out_name_ = name; out_count_ = n;
    }
  }
  if (engine_->getNbIOTensors() != 2 || in_name_.empty() || out_name_.empty())
    throw std::runtime_error("week 1 supports exactly 1 input / 1 output");

  // 4) GPU 버퍼 1회 할당 + 주소를 context에 한 번만 등록
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUDA_CHECK(cudaMalloc(&d_in_, in_count_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_, out_count_ * sizeof(float)));
  context_->setTensorAddress(in_name_.c_str(), d_in_);
  context_->setTensorAddress(out_name_.c_str(), d_out_);

  std::cerr << "loaded " << engine_path << "  in=" << in_name_ << "(" << in_count_
            << ")  out=" << out_name_ << "(" << out_count_ << ")\n";
}

Runner::~Runner() {
  // 소멸자에서는 예외를 던지지 않는다.
  if (d_in_) cudaFree(d_in_);
  if (d_out_) cudaFree(d_out_);
  if (stream_) cudaStreamDestroy(stream_);
}

void Runner::infer(const float* in, float* out) {
  // 모두 같은 stream에 비동기로 올리고 마지막에 한 번만 동기화한다.
  CUDA_CHECK(cudaMemcpyAsync(d_in_, in, in_count_ * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));
  if (!context_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
  CUDA_CHECK(cudaMemcpyAsync(out, d_out_, out_count_ * sizeof(float),
                             cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace minirt
