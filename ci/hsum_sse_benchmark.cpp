#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ducc0/math/gridding_kernel.h"

#if !defined(__SSE2__) || defined(__AVX__)
#error "compile this benchmark for SSE2 with AVX disabled"
#endif
#if defined(DUCC_HSUM_EXPECT_SSE3)
#if !defined(__SSE3__)
#error "the SSE3 benchmark target did not define __SSE3__"
#endif
#elif defined(DUCC_HSUM_EXPECT_SSE2)
#if defined(__SSE3__)
#error "the SSE2-only benchmark target unexpectedly defined __SSE3__"
#endif
#else
#error "define DUCC_HSUM_EXPECT_SSE3 or DUCC_HSUM_EXPECT_SSE2"
#endif
#ifdef DUCC0_NO_SIMD
#error "DUCC0_NO_SIMD must not be active"
#endif

#if defined(_MSC_VER)
#define DUCC_BENCH_NOINLINE __declspec(noinline)
#else
#define DUCC_BENCH_NOINLINE __attribute__((noinline))
#endif

namespace {

#if defined(DUCC_HSUM_EXPECT_SSE3)
constexpr const char *target_name = "SSE3/no-AVX";
#else
constexpr const char *target_name = "SSE2-only/no-AVX";
#endif

using vec = ducc0::bounded_simd<float,8>;
static_assert(vec::size()==4, "the SSE benchmark must use four floats");

volatile float sink = 0.f;

std::uint32_t next_state(std::uint32_t &state)
  {
  state = state*1664525u + 1013904223u;
  return state;
  }

void make_input(std::vector<float> &real, std::vector<float> &imag)
  {
  std::uint32_t state = 0x20260914u;
  for (size_t i=0; i<real.size(); ++i)
    {
    real[i] = static_cast<float>((next_state(state)>>8)*0x1p-24f - .5f);
    imag[i] = static_cast<float>((next_state(state)>>8)*0x1p-24f - .5f);
    }
  }

struct stats
  {
  double median, minimum, maximum, stdev;
  };

stats summarize(std::vector<double> values)
  {
  std::sort(values.begin(), values.end());
  double mean = 0.;
  for (auto value: values) mean += value;
  mean /= values.size();
  double variance = 0.;
  for (auto value: values) variance += (value-mean)*(value-mean);
  variance /= values.size();
  const size_t n = values.size();
  return {(n&1) ? values[n/2] : .5*(values[n/2-1]+values[n/2]),
    values.front(), values.back(), std::sqrt(variance)};
  }

DUCC_BENCH_NOINLINE double run_benchmark(size_t iterations)
  {
  vec real[8], imag[8];
  float state_r[8], state_i[8];
  for (size_t j=0; j<8; ++j)
    {
    real[j] = vec(.01f+.001f*static_cast<float>(j));
    imag[j] = vec(.02f-.001f*static_cast<float>(j));
    state_r[j] = .0001f+.00001f*static_cast<float>(j);
    state_i[j] = -.0002f-.00001f*static_cast<float>(j);
    }
  const auto start = std::chrono::steady_clock::now();
  for (size_t i=0; i<iterations; ++i)
    for (size_t j=0; j<8; ++j)
      {
      real[j] += vec(1.e-5f*state_r[j]);
      imag[j] += vec(1.e-5f*state_i[j]);
      const auto z = ducc0::detail_gridding_kernel::hsum_cmplx<float>(
        real[j], imag[j]);
      state_r[j] = .999f*state_r[j] + 1.e-6f*(z.real()-.5f*z.imag());
      state_i[j] = .999f*state_i[j] + 1.e-6f*(z.imag()+.5f*z.real());
      }
  const auto stop = std::chrono::steady_clock::now();
  float local_sink = 0.f;
  for (size_t j=0; j<8; ++j) local_sink += state_r[j]+state_i[j];
  sink = local_sink;
  const auto elapsed = std::chrono::duration<double, std::nano>(stop-start).count();
  return elapsed/(static_cast<double>(iterations)*8.);
  }

}  // namespace

int main(int argc, char **argv)
  {
  if (argc != 5)
    {
    std::cerr << "usage: hsum_sse_benchmark iterations warmups repetitions output_dir\n";
    return 2;
    }
  const size_t iterations = std::stoull(argv[1]);
  const size_t warmups = std::stoull(argv[2]);
  const size_t repetitions = std::stoull(argv[3]);
  const std::string output_dir = argv[4];
  if ((iterations==0) || (repetitions==0)) return 2;

  std::vector<float> real(4096*4), imag(4096*4);
  make_input(real, imag);
  std::ofstream values(output_dir+"/values.bin", std::ios::binary);
  if (!values) return 3;
  for (size_t i=0; i<4096; ++i)
    {
    const auto r = ducc0::detail_gridding_kernel::hsum_cmplx<float>(
      ducc0::loadu<vec>(&real[4*i]), ducc0::loadu<vec>(&imag[4*i]));
    const float result[2] = {r.real(), r.imag()};
    values.write(reinterpret_cast<const char *>(result), sizeof(result));
    }
  values.close();

  for (size_t i=0; i<warmups; ++i) run_benchmark(iterations);
  std::vector<double> samples;
  samples.reserve(repetitions);
  for (size_t i=0; i<repetitions; ++i) samples.push_back(run_benchmark(iterations));
  const auto result = summarize(samples);

  std::ofstream report(output_dir+"/stats.json");
  if (!report) return 3;
  report << std::setprecision(17)
    << "{\"target\":\"" << target_name << "\",\"median_ns\":" << result.median
    << ",\"minimum_ns\":" << result.minimum
    << ",\"maximum_ns\":" << result.maximum
    << ",\"stdev_ns\":" << result.stdev
    << ",\"sink\":" << sink << "}\n";
  std::cout << std::fixed << std::setprecision(6)
    << target_name << " median_ns=" << result.median
    << " minimum_ns=" << result.minimum
    << " maximum_ns=" << result.maximum
    << " stdev_ns=" << result.stdev
    << " sink=" << sink << '\n';
  }
