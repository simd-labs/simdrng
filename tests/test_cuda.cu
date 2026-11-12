// Device/host equality for the five scalar generators, run without
// --expt-relaxed-constexpr. One kernel per generator with 256 threads; thread i
// builds the generator from the seed and applies i jump() calls (XoshiroScalar,
// MWC64X32) or sets the counter word to i (Philox, ChaCha), draws 16 values,
// then runs one long_jump() and 4 more draws where the class has long_jump(),
// and one uniform() where the class has it. The host recomputes every thread's
// sequence with the same class and the test requires bit equality.
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <simdrng/chacha.hpp>
#include <simdrng/mwc64x32.hpp>
#include <simdrng/philox.hpp>
#include <simdrng/splitmix.hpp>
#include <simdrng/xoshiro_scalar.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kDraws = 16;
constexpr int kPost = 4;
constexpr int kOut = kDraws + kPost;

using seq_fn = void (*)(std::uint64_t, int, std::uint64_t *, double *);

// ChaCha key derived from the seed with the same code on both sides.
SIMDRNG_HD simdrng::array<std::uint32_t, 8> chacha_key(std::uint64_t seed) {
  simdrng::SplitMix sm{seed};
  simdrng::array<std::uint32_t, 8> key{};
  for (std::size_t w = 0; w < 8; ++w) {
    key[w] = static_cast<std::uint32_t>(sm());
  }
  return key;
}

// SplitMix has no jump() or counter, so every thread draws the seed stream.
SIMDRNG_HD void seq_splitmix(std::uint64_t seed, int i, std::uint64_t *out, double *unif) {
  (void)i;
  simdrng::SplitMix g{seed};
  for (int k = 0; k < kOut; ++k) {
    out[k] = g();
  }
  *unif = 0.0; // SplitMix has no uniform()
}

SIMDRNG_HD void seq_xoshiro(std::uint64_t seed, int i, std::uint64_t *out, double *unif) {
  simdrng::XoshiroScalar g{seed};
  for (int j = 0; j < i; ++j) {
    g.jump();
  }
  for (int k = 0; k < kDraws; ++k) {
    out[k] = g();
  }
  g.long_jump();
  for (int k = 0; k < kPost; ++k) {
    out[kDraws + k] = g();
  }
  *unif = g.uniform();
}

SIMDRNG_HD void seq_mwc(std::uint64_t seed, int i, std::uint64_t *out, double *unif) {
  simdrng::MWC64X32 g{seed};
  for (int j = 0; j < i; ++j) {
    g.jump();
  }
  for (int k = 0; k < kDraws; ++k) {
    out[k] = g();
  }
  g.long_jump();
  for (int k = 0; k < kPost; ++k) {
    out[kDraws + k] = g();
  }
  *unif = 0.0; // MWC64X32 has no uniform()
}

SIMDRNG_HD void seq_philox(std::uint64_t seed, int i, std::uint64_t *out, double *unif) {
  simdrng::Philox<> g{seed};
  g.setCounter(simdrng::Philox<>::counter_from_uint64(static_cast<std::uint64_t>(i)));
  for (int k = 0; k < kDraws; ++k) {
    out[k] = g();
  }
  // Philox has no long_jump(); the extra draws continue the counter stream.
  for (int k = 0; k < kPost; ++k) {
    out[kDraws + k] = g();
  }
  *unif = g.uniform();
}

SIMDRNG_HD void seq_chacha(std::uint64_t seed, int i, std::uint64_t *out, double *unif) {
  const auto key = chacha_key(seed);
  // The ctor keeps its std::array key parameter (as at 85cc841).
  simdrng::ChaCha<> g{simdrng_bit_cast<std::array<std::uint32_t, 8>>(key), static_cast<std::uint64_t>(i), 0};
  for (int k = 0; k < kDraws; ++k) {
    out[k] = g();
  }
  // ChaCha has no long_jump(); the extra draws continue the counter stream.
  for (int k = 0; k < kPost; ++k) {
    out[kDraws + k] = g();
  }
  *unif = g.uniform();
}

__global__ void k_splitmix(std::uint64_t seed, std::uint64_t *out, double *unif) {
  const auto i = static_cast<int>(threadIdx.x);
  seq_splitmix(seed, i, out + i * kOut, unif + i);
}
__global__ void k_xoshiro(std::uint64_t seed, std::uint64_t *out, double *unif) {
  const auto i = static_cast<int>(threadIdx.x);
  seq_xoshiro(seed, i, out + i * kOut, unif + i);
}
__global__ void k_mwc(std::uint64_t seed, std::uint64_t *out, double *unif) {
  const auto i = static_cast<int>(threadIdx.x);
  seq_mwc(seed, i, out + i * kOut, unif + i);
}
__global__ void k_philox(std::uint64_t seed, std::uint64_t *out, double *unif) {
  const auto i = static_cast<int>(threadIdx.x);
  seq_philox(seed, i, out + i * kOut, unif + i);
}
__global__ void k_chacha(std::uint64_t seed, std::uint64_t *out, double *unif) {
  const auto i = static_cast<int>(threadIdx.x);
  seq_chacha(seed, i, out + i * kOut, unif + i);
}

using kernel_fn = void (*)(std::uint64_t, std::uint64_t *, double *);

// Launch the kernel, check both CUDA error paths, copy back, compare every
// value with the host-side sequence.
void check_equal(kernel_fn kernel, seq_fn seq, std::uint64_t seed) {
  std::uint64_t *dev_out = nullptr;
  double *dev_unif = nullptr;
  REQUIRE(cudaMalloc(&dev_out, static_cast<std::size_t>(kThreads) * kOut * sizeof(std::uint64_t)) == cudaSuccess);
  REQUIRE(cudaMalloc(&dev_unif, static_cast<std::size_t>(kThreads) * sizeof(double)) == cudaSuccess);

  kernel<<<1, kThreads>>>(seed, dev_out, dev_unif);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::uint64_t host_out[static_cast<std::size_t>(kThreads) * kOut];
  double host_unif[kThreads];
  REQUIRE(cudaMemcpy(host_out, dev_out, sizeof(host_out), cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(host_unif, dev_unif, sizeof(host_unif), cudaMemcpyDeviceToHost) == cudaSuccess);
  cudaFree(dev_out);
  cudaFree(dev_unif);

  std::uint64_t ref[kOut];
  double ref_unif = 0.0;
  int bad = 0;
  for (int i = 0; i < kThreads; ++i) {
    seq(seed, i, ref, &ref_unif);
    for (int k = 0; k < kOut; ++k) {
      if (host_out[static_cast<std::size_t>(i) * kOut + k] != ref[k]) {
        ++bad;
      }
    }
    if (std::memcmp(&host_unif[i], &ref_unif, sizeof(double)) != 0) {
      ++bad;
    }
  }
  CHECK(bad == 0);
}

// Hierarchical streams: 4096 threads, each builds Gen{seed}.stream(levels...)
// from its gid (node = gid & 3, gpu = gid >> 8 or gid >> 10, thread = gid &
// 0xFF or gid & 0x3FF) and draws 8 values. The host runs the same sseq
// functions, so bit equality covers the host/device parity of the level paths.
// All generators accept node.
constexpr int kStreamThreads = 4096;
constexpr int kStreamDraws = 8;

// SplitMix takes node too: gpu/node/thread in a non-default order on purpose.
SIMDRNG_HD void sseq_splitmix(std::uint64_t seed, std::uint64_t gid, std::uint64_t *out) {
  auto g = simdrng::SplitMix{seed}.stream(simdrng::gpu{gid >> 10}, simdrng::node{gid & 3},
                                          simdrng::thread{gid & 0x3FF});
  for (int k = 0; k < kStreamDraws; ++k) {
    out[k] = g();
  }
}

SIMDRNG_HD void sseq_xoshiro(std::uint64_t seed, std::uint64_t gid, std::uint64_t *out) {
  auto g = simdrng::XoshiroScalar{seed}.stream(simdrng::node{gid & 3}, simdrng::gpu{gid >> 10},
                                               simdrng::thread{gid & 0x3FF});
  for (int k = 0; k < kStreamDraws; ++k) {
    out[k] = g();
  }
}

SIMDRNG_HD void sseq_mwc(std::uint64_t seed, std::uint64_t gid, std::uint64_t *out) {
  auto g = simdrng::MWC64X32{seed}.stream(simdrng::gpu{gid >> 10}, simdrng::node{gid & 3}, simdrng::thread{gid & 0x3FF});
  for (int k = 0; k < kStreamDraws; ++k) {
    out[k] = g();
  }
}

SIMDRNG_HD void sseq_philox(std::uint64_t seed, std::uint64_t gid, std::uint64_t *out) {
  // Reverse argument order on purpose: order must not change the stream.
  auto g = simdrng::Philox<>{seed}.stream(simdrng::thread{gid & 0xFF}, simdrng::gpu{gid >> 8}, simdrng::node{gid & 3});
  for (int k = 0; k < kStreamDraws; ++k) {
    out[k] = g();
  }
}

// The stream F3 broke: Philox2x32 gives one output per block and its index
// used to fill the whole counter.
SIMDRNG_HD void sseq_philox2x32(std::uint64_t seed, std::uint64_t gid, std::uint64_t *out) {
  auto g =
      simdrng::Philox2x32{seed}.stream(simdrng::gpu{gid >> 8}, simdrng::thread{gid & 0xFF}, simdrng::node{gid & 3});
  for (int k = 0; k < kStreamDraws; ++k) {
    out[k] = g();
  }
}

SIMDRNG_HD void sseq_chacha(std::uint64_t seed, std::uint64_t gid, std::uint64_t *out) {
  auto g = simdrng::ChaCha<>{simdrng_bit_cast<std::array<std::uint32_t, 8>>(chacha_key(seed)), 0, 0}.stream(
      simdrng::gpu{gid >> 8}, simdrng::node{gid & 3}, simdrng::thread{gid & 0xFF});
  for (int k = 0; k < kStreamDraws; ++k) {
    out[k] = g();
  }
}

__global__ void ks_splitmix(std::uint64_t seed, std::uint64_t *out) {
  const auto gid = static_cast<std::uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  sseq_splitmix(seed, gid, out + gid * kStreamDraws);
}
__global__ void ks_xoshiro(std::uint64_t seed, std::uint64_t *out) {
  const auto gid = static_cast<std::uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  sseq_xoshiro(seed, gid, out + gid * kStreamDraws);
}
__global__ void ks_mwc(std::uint64_t seed, std::uint64_t *out) {
  const auto gid = static_cast<std::uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  sseq_mwc(seed, gid, out + gid * kStreamDraws);
}
__global__ void ks_philox(std::uint64_t seed, std::uint64_t *out) {
  const auto gid = static_cast<std::uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  sseq_philox(seed, gid, out + gid * kStreamDraws);
}
__global__ void ks_philox2x32(std::uint64_t seed, std::uint64_t *out) {
  const auto gid = static_cast<std::uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  sseq_philox2x32(seed, gid, out + gid * kStreamDraws);
}
__global__ void ks_chacha(std::uint64_t seed, std::uint64_t *out) {
  const auto gid = static_cast<std::uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  sseq_chacha(seed, gid, out + gid * kStreamDraws);
}

using skernel_fn = void (*)(std::uint64_t, std::uint64_t *);
using sseq_fn = void (*)(std::uint64_t, std::uint64_t, std::uint64_t *);

// Launch the 4096-thread stream kernel, check both CUDA error paths, compare
// every thread's 8 values with the host-side stream.
void check_streams(skernel_fn kernel, sseq_fn seq, std::uint64_t seed) {
  std::uint64_t *dev_out = nullptr;
  REQUIRE(cudaMalloc(&dev_out, static_cast<std::size_t>(kStreamThreads) * kStreamDraws * sizeof(std::uint64_t)) ==
          cudaSuccess);

  kernel<<<kStreamThreads / 256, 256>>>(seed, dev_out);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  std::vector<std::uint64_t> host(static_cast<std::size_t>(kStreamThreads) * kStreamDraws);
  REQUIRE(cudaMemcpy(host.data(), dev_out, host.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost) == cudaSuccess);
  cudaFree(dev_out);

  std::uint64_t ref[kStreamDraws];
  int bad = 0;
  for (int i = 0; i < kStreamThreads; ++i) {
    seq(seed, static_cast<std::uint64_t>(i), ref);
    for (int k = 0; k < kStreamDraws; ++k) {
      if (host[static_cast<std::size_t>(i) * kStreamDraws + k] != ref[k]) {
        ++bad;
      }
    }
  }
  CHECK(bad == 0);
}

} // namespace

// Device rotl must equal host rotl for the same (value, k) cases. Thread i
// rotates x32/x64 (derived from i) at k = 0, 1, width-1, and a mid value, and
// the host recomputes with the same simdrng::rotl.
constexpr int kRotlThreads = 64;

__global__ void k_rotl(std::uint64_t *out) {
  const auto i = static_cast<int>(threadIdx.x);
  const std::uint32_t x32 = 0xDEADBEEFu + static_cast<std::uint32_t>(i);
  const std::uint64_t x64 = 0xDEADBEEFCAFEBABEull + static_cast<std::uint64_t>(i);
  std::uint64_t *o = out + i * 8;
  o[0] = simdrng::rotl(x32, 0);
  o[1] = simdrng::rotl(x32, 1);
  o[2] = simdrng::rotl(x32, 31);
  o[3] = simdrng::rotl(x32, 10);
  o[4] = simdrng::rotl(x64, 0);
  o[5] = simdrng::rotl(x64, 1);
  o[6] = simdrng::rotl(x64, 63);
  o[7] = simdrng::rotl(x64, 21);
}

TEST_CASE("rotl device output equals host", "[cuda]") {
  std::uint64_t *dev_out = nullptr;
  REQUIRE(cudaMalloc(&dev_out, static_cast<std::size_t>(kRotlThreads) * 8 * sizeof(std::uint64_t)) == cudaSuccess);
  k_rotl<<<1, kRotlThreads>>>(dev_out);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  std::vector<std::uint64_t> host(static_cast<std::size_t>(kRotlThreads) * 8);
  REQUIRE(cudaMemcpy(host.data(), dev_out, host.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost) == cudaSuccess);
  cudaFree(dev_out);
  int bad = 0;
  const int ks32[] = {0, 1, 31, 10};
  const int ks64[] = {0, 1, 63, 21};
  for (int i = 0; i < kRotlThreads; ++i) {
    const std::uint32_t x32 = 0xDEADBEEFu + static_cast<std::uint32_t>(i);
    const std::uint64_t x64 = 0xDEADBEEFCAFEBABEull + static_cast<std::uint64_t>(i);
    for (int k = 0; k < 4; ++k) {
      if (host[static_cast<std::size_t>(i) * 8 + k] != simdrng::rotl(x32, ks32[k]) ||
          host[static_cast<std::size_t>(i) * 8 + 4 + k] != simdrng::rotl(x64, ks64[k])) {
        ++bad;
      }
    }
  }
  CHECK(bad == 0);
}

TEST_CASE("SplitMix device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_equal(k_splitmix, seq_splitmix, seed);
}

TEST_CASE("XoshiroScalar device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_equal(k_xoshiro, seq_xoshiro, seed);
}

TEST_CASE("MWC64X32 device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_equal(k_mwc, seq_mwc, seed);
}

TEST_CASE("Philox device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_equal(k_philox, seq_philox, seed);
}

TEST_CASE("ChaCha device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_equal(k_chacha, seq_chacha, seed);
}

TEST_CASE("SplitMix stream device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_streams(ks_splitmix, sseq_splitmix, seed);
}

TEST_CASE("XoshiroScalar stream device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_streams(ks_xoshiro, sseq_xoshiro, seed);
}

TEST_CASE("MWC64X32 stream device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_streams(ks_mwc, sseq_mwc, seed);
}

TEST_CASE("Philox stream device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_streams(ks_philox, sseq_philox, seed);
}

TEST_CASE("Philox2x32 stream device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_streams(ks_philox2x32, sseq_philox2x32, seed);
}

TEST_CASE("ChaCha stream device output equals host", "[cuda]") {
  const auto seed = GENERATE(std::uint64_t{0x0102030405060708ULL}, std::uint64_t{0xdeadbeefcafebabeULL});
  check_streams(ks_chacha, sseq_chacha, seed);
}
