# jetson-mini-runtime

Jetson Xavier NX에서 ONNX 모델을 실행하는 **C++ 미니 추론 런타임**(TensorRT 백엔드)과 측정 도구입니다.
엔진 로드, 버퍼 관리, 실행, 측정을 직접 구현하고, 공식 도구(trtexec)와 비교하며 런타임 오버헤드를 분석합니다.

## 환경
| 항목 | 버전 |
|---|---|
| 보드 | Jetson Xavier NX (RAM 8GB, CPU·GPU 공유) |
| JetPack / L4T | 5.1.5 / R35.6.4 |
| CUDA / TensorRT | 11.4 / 8.5.2 |
| 컴파일러 / CMake | g++ 9.4 (C++17) / 3.26 |
| 측정 조건 | `nvpmodel -m 8` (MODE_20W_6CORE) + `jetson_clocks` |

## 1주차 결과 (ResNet-18, FP32, batch 1)
| 도구 | p50 (ms) | p95 (ms) | p99 (ms) | FPS |
|---|---|---|---|---|
| trtexec (기준치) | 4.956 | 4.995 | 5.017 | 203.5 |
| **mini-runtime** (3,000프레임, 워밍업 100) | 5.155 | 5.236 | 5.311 | 193.7 |

- **정확도:** PyTorch 출력과 비교해 top-1 일치, 최대 오차 4.6e-6
- **차이 분석 (약 0.2ms):** GPU 연산은 같은 엔진이라 동일합니다. trtexec의 Latency는 GPU 이벤트 기준이고 다음 요청을 미리 enqueue하지만,
  mini-runtime은 CPU 시계로 `infer()` 전체를 측정해 enqueue(약 0.38ms)와 동기화 대기, pageable 메모리 복사 비용이 포함됩니다.

## 구조
```
include/minirt/runner.h   엔진 로드 + GPU 버퍼 1회 할당 + infer()
src/runner.cpp
tools/run_once.cpp        1회 추론 + PyTorch 기준값과 정확도 비교
tools/bench.cpp           N프레임 측정 → p50/p95/p99, FPS → results/bench.csv
scripts/export_onnx.py    PyTorch → ONNX (opset 13) + 기준 입출력 저장
results/                  측정 결과 (CSV, trtexec 로그)
```

## 설계 결정
- **빌드와 실행 분리:** 엔진 빌드는 수십 초가 걸리고 기기·TRT 버전에 종속되므로 오프라인 단계로 분리했습니다.
  런타임은 `.engine`만 읽어서 ONNX 파서를 링크하지 않습니다.
- **버퍼 1회 할당:** GPU 버퍼와 텐서 주소 등록은 생성자에서 한 번만 하고, 매 프레임에는 할당하지 않습니다 (RAII로 해제).
- **이름 기반 I/O (TRT 8.5 API):** `getIOTensorName` → `setTensorAddress` → `enqueueV3`. 바인딩 인덱스 방식은 deprecated입니다.
- **비동기 실행 + 동기화 1회:** H2D 복사, 실행, D2H 복사를 같은 스트림에 넣고 마지막에 한 번만 동기화합니다.
- **측정과 실행 분리:** 측정 코드는 Runner 밖(`bench`)에 두어, 이후 파이프라인에서 Runner를 그대로 재사용합니다.
- **정확도 먼저:** 속도 측정 전에 PyTorch 출력과 일치하는지 검증합니다.

## 실행 방법
```bash
python3 scripts/export_onnx.py
/usr/src/tensorrt/bin/trtexec --onnx=models/resnet18.onnx \
  --saveEngine=models/resnet18_fp32.engine --memPoolSize=workspace:512

cmake -S . -B build && cmake --build build -j4
./build/run_once models/resnet18_fp32.engine models/ref_input.npy models/ref_output.npy

sudo nvpmodel -m 8 && sudo jetson_clocks
./build/bench models/resnet18_fp32.engine --precision fp32 --power MODE_20W_6CORE
```

## 로드맵
- [x] 1주차: ONNX 변환, TensorRT 런타임 뼈대, 정확도 검증, 지연·처리량 측정
- [ ] 1주차: 자체 엔진 빌더(`build_engine`)로 trtexec 대체
- [ ] 2주차: 메모리 플래닝 — 텐서 생존 구간 분석 후 버퍼 재사용, TRT 결과(4.8MB)와 비교
- [ ] 3주차: 전처리/추론/후처리 멀티스레드 파이프라인, FP32/FP16 비교
- [ ] 4주차: PC↔Jetson 원격 벤치마크, `.so` 분석 및 용량 축소

## 개발 방식
설계 방향은 직접 정하고, 보일러플레이트 코드 초안은 AI 코딩 도구(Claude)로 작성했습니다.
모든 코드는 Jetson에서 직접 빌드·실행해 검증했고, 결정 근거는 `docs/decisions.md`에 기록합니다.
