// 지연/처리량 측정: 워밍업 후 N프레임 반복, p50/p95/p99 + FPS, CSV에 한 줄 추가.
// 사용법: bench <engine> [--frames 3000] [--warmup 100]
//              [--precision fp32] [--power MODE_20W_6CORE] [--csv results/bench.csv]
//
// 설계 이유
// - Runner는 그대로 두고 밖에서 잰다(측정과 실행 분리, Q3).
// - 측정 구간 = infer() 1회 전체(H2D + 실행 + D2H + 동기화) = trtexec의 "Latency"와 같은 범위.
//   그래서 trtexec 결과와 바로 비교할 수 있다.
// - 워밍업: 첫 실행들은 GPU 클럭 상승, 커널 로드, 캐시 때문에 느리다. 통계에서 제외.
// - steady_clock: 시스템 시간 변경에 영향받지 않는 단조 증가 시계.
// - 측정 조건(정밀도, 전원 모드, 배치)을 결과와 같은 줄에 기록 -> 주차별 비교 가능.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "minirt/runner.h"

using Clock = std::chrono::steady_clock;

// nearest-rank 백분위수: 정렬된 배열에서 ceil(p*n)번째 값.
static double percentile(const std::vector<double>& sorted, double p) {
  size_t idx = static_cast<size_t>(std::ceil(p * sorted.size()));
  if (idx == 0) idx = 1;
  return sorted[idx - 1];
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: bench <engine> [--frames N] [--warmup N] [--precision S]"
                 " [--power S] [--csv PATH]\n";
    return 1;
  }
  std::string engine = argv[1], precision = "fp32", power = "unknown",
              csv = "results/bench.csv";
  int frames = 3000, warmup = 100;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--frames") frames = std::stoi(v);
    else if (k == "--warmup") warmup = std::stoi(v);
    else if (k == "--precision") precision = v;
    else if (k == "--power") power = v;
    else if (k == "--csv") csv = v;
  }

  try {
    minirt::Runner r(engine);
    std::vector<float> in(r.inputCount(), 0.5f), out(r.outputCount());

    for (int i = 0; i < warmup; ++i) r.infer(in.data(), out.data());

    std::vector<double> lat_ms(frames);
    auto t0 = Clock::now();
    for (int i = 0; i < frames; ++i) {
      auto s = Clock::now();
      r.infer(in.data(), out.data());
      auto e = Clock::now();
      lat_ms[i] = std::chrono::duration<double, std::milli>(e - s).count();
    }
    double total_s = std::chrono::duration<double>(Clock::now() - t0).count();

    std::vector<double> sorted = lat_ms;
    std::sort(sorted.begin(), sorted.end());
    double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / frames;
    double p50 = percentile(sorted, 0.50), p95 = percentile(sorted, 0.95),
           p99 = percentile(sorted, 0.99);
    double fps = frames / total_s;

    std::cout << "frames=" << frames << " warmup=" << warmup
              << " precision=" << precision << " batch=1 power=" << power << "\n"
              << "latency ms: mean=" << mean << " p50=" << p50 << " p95=" << p95
              << " p99=" << p99 << " min=" << sorted.front()
              << " max=" << sorted.back() << "\n"
              << "FPS=" << fps << "\n";

    bool new_file = !std::ifstream(csv).good();
    std::ofstream f(csv, std::ios::app);
    if (!f) throw std::runtime_error("cannot write " + csv);
    if (new_file)
      f << "timestamp,engine,precision,batch,power,frames,warmup,"
           "mean_ms,p50_ms,p95_ms,p99_ms,min_ms,max_ms,fps\n";
    std::time_t now = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    f << ts << "," << engine << "," << precision << ",1," << power << "," << frames
      << "," << warmup << "," << mean << "," << p50 << "," << p95 << "," << p99
      << "," << sorted.front() << "," << sorted.back() << "," << fps << "\n";
    std::cout << "appended to " << csv << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
