// 1회 추론 + PyTorch 기준값과 비교.
// 사용법: run_once <engine> <ref_input.npy> <ref_output.npy>
//
// 설계 이유: 속도를 재기 전에 "결과가 맞는지"부터 확인한다.
// 틀린 결과를 빠르게 내는 런타임은 의미가 없다.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "minirt/runner.h"

// 최소 .npy 로더: float32, little-endian, C-order만 지원.
static std::vector<float> loadNpy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
  if (b.size() < 10 || std::memcmp(b.data(), "\x93NUMPY", 6) != 0)
    throw std::runtime_error("not npy: " + path);
  size_t hlen, start;
  if (b[6] == 1) { hlen = uint8_t(b[8]) | (uint8_t(b[9]) << 8); start = 10 + hlen; }
  else { uint32_t h; std::memcpy(&h, &b[8], 4); hlen = h; start = 12 + hlen; }
  std::string header(b.begin() + (start - hlen), b.begin() + start);
  if (header.find("<f4") == std::string::npos ||
      header.find("'fortran_order': False") == std::string::npos)
    throw std::runtime_error("need float32 C-order npy: " + path);
  std::vector<float> v((b.size() - start) / sizeof(float));
  std::memcpy(v.data(), b.data() + start, v.size() * sizeof(float));
  return v;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: run_once <engine> <ref_input.npy> <ref_output.npy>\n";
    return 1;
  }
  try {
    minirt::Runner r(argv[1]);
    auto in = loadNpy(argv[2]);
    auto ref = loadNpy(argv[3]);
    if (in.size() != r.inputCount() || ref.size() != r.outputCount())
      throw std::runtime_error("size mismatch between npy and engine");

    std::vector<float> out(r.outputCount());
    r.infer(in.data(), out.data());

    float max_diff = 0.f;
    for (size_t i = 0; i < out.size(); ++i)
      max_diff = std::max(max_diff, std::fabs(out[i] - ref[i]));
    auto top = [](const std::vector<float>& v) {
      return int(std::max_element(v.begin(), v.end()) - v.begin());
    };
    std::cout << "top-1  trt=" << top(out) << "  torch=" << top(ref) << "\n"
              << "max |trt - torch| = " << max_diff << "\n"
              << (top(out) == top(ref) ? "PASS" : "FAIL") << "\n";
    return top(out) == top(ref) ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
