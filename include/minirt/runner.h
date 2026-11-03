#pragma once
// Runner: 직렬화된 TensorRT 엔진을 로드하고 추론을 실행한다.
//
// 설계 이유
// - 빌드와 분리(Q1): 이 클래스는 .engine만 읽는다. ONNX 파서에 의존하지 않는다.
// - 버퍼 1회 할당(Q2): GPU 버퍼는 생성자에서 한 번만 cudaMalloc, 소멸자에서 해제(RAII).
// - 측정과 분리(Q3): 시간 측정 코드는 여기 없다. 호출하는 쪽(bench)이 잰다.
// - 1주차 범위: 입력 1개 / 출력 1개 / FP32 I/O / 정적 shape만 지원. 아니면 예외.

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>

namespace minirt {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

class Runner {
 public:
  explicit Runner(const std::string& engine_path);
  ~Runner();
  Runner(const Runner&) = delete;             // GPU 버퍼 소유권 중복 방지
  Runner& operator=(const Runner&) = delete;

  // H2D 복사 -> enqueueV3 -> D2H 복사 -> 동기화. 반환 시 out이 채워져 있다.
  void infer(const float* in, float* out);

  size_t inputCount() const { return in_count_; }    // 원소 개수
  size_t outputCount() const { return out_count_; }

 private:
  // 선언 순서 = 생성 순서, 소멸은 역순 (context -> engine -> runtime).
  // TensorRT는 이 순서로 해제해야 안전하다.
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  cudaStream_t stream_ = nullptr;
  void* d_in_ = nullptr;
  void* d_out_ = nullptr;
  size_t in_count_ = 0, out_count_ = 0;
  std::string in_name_, out_name_;
};

}  // namespace minirt
