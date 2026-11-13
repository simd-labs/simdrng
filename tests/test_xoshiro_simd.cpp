#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include <catch2/catch_all.hpp>
#include <simdrng/xoshiro_simd.hpp>

#include "stream_levels_contract.hpp"

static constexpr auto tests = 1 << 12; // 4096

// The tag layout lane stride S = 2^(127 - ceil(log2(W))). The exponent comes
// from the spec formula here, not from the library, so the layout tests fail
// when the library moves the lanes.
static std::uint64_t lane_stride_exp(const std::uint8_t W) {
  std::uint64_t lg = 0;
  while ((std::uint64_t{1} << lg) < W) {
    ++lg;
  }
  return 127 - lg;
}

// Lane i of the tagged generator must equal the scalar generator at the block
// start advanced by i * S. The advance uses the pow2 jump helpers.
template <class Gen, typename... Levels>
static void check_tagged_lanes(const std::uint8_t W, const std::uint64_t seed, Levels... levels) {
  const Gen g{seed, levels...};
  std::array<std::uint64_t, 32> flat{};
  g.get_flat_state(flat.data());
  const std::uint64_t k = lane_stride_exp(W);
  for (std::uint8_t i = 0; i < W; ++i) {
    simdrng::XoshiroScalar lane{seed, levels...};
    for (std::uint64_t b = 0; (std::uint64_t{1} << b) <= i; ++b) {
      if (((i >> b) & 1U) != 0U) {
        lane.jump(simdrng::pow2{k + b});
      }
    }
    const auto state = lane.getState();
    for (std::uint8_t r = 0; r < 4; ++r) {
      INFO("lane " << static_cast<unsigned>(i) << " reg " << static_cast<unsigned>(r));
      REQUIRE(flat[r * W + i] == state[r]);
    }
  }
  // Lane 0 draws the scalar stream output by output. The SIMD generator
  // interleaves the lanes in its output order, so lane 0 contributes every
  // W-th value.
  Gen out{seed, levels...};
  simdrng::XoshiroScalar s0{seed, levels...};
  for (int j = 0; j < 64; ++j) {
    INFO("draw " << j);
    const std::uint64_t want = s0();
    std::uint64_t got = 0;
    for (std::uint8_t k = 0; k < W; ++k) {
      const std::uint64_t v = out();
      if (k == 0) {
        got = v;
      }
    }
    REQUIRE(got == want);
  }
}

TEST_CASE("TAG LANE LAYOUT", "[xoshiro256++][stream]") {
  const std::uint64_t seed = 42;
  check_tagged_lanes<simdrng::XoshiroSIMD>(simdrng::XoshiroSIMD{seed}.simd_width(), seed, simdrng::node{1},
                                           simdrng::gpu{2}, simdrng::thread{3});
  check_tagged_lanes<simdrng::XoshiroSIMD>(simdrng::XoshiroSIMD{seed}.simd_width(), seed, simdrng::thread{3});
  check_tagged_lanes<simdrng::XoshiroSIMD>(simdrng::XoshiroSIMD{seed}.simd_width(), seed, simdrng::gpu{2});
  check_tagged_lanes<simdrng::XoshiroSIMD>(simdrng::XoshiroSIMD{seed}.simd_width(), seed, simdrng::node{7});
  check_tagged_lanes<simdrng::XoshiroSIMD>(simdrng::XoshiroSIMD{seed}.simd_width(), seed, simdrng::thread{5},
                                           simdrng::node{3});
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  constexpr auto W = simdrng::XoshiroNative::simd_width();
  check_tagged_lanes<simdrng::XoshiroNative>(W, seed, simdrng::node{1}, simdrng::gpu{2}, simdrng::thread{3});
  check_tagged_lanes<simdrng::XoshiroNative>(W, seed, simdrng::thread{3});
  check_tagged_lanes<simdrng::XoshiroNative>(W, seed, simdrng::gpu{2});
  check_tagged_lanes<simdrng::XoshiroNative>(W, seed, simdrng::node{7});
  check_tagged_lanes<simdrng::XoshiroNative>(W, seed, simdrng::thread{5}, simdrng::node{3});
#endif
}

TEST_CASE("TAG LEVEL CONTRACT", "[xoshiro256++][stream]") {
  const std::uint64_t seed = 42;
  const auto make_simd = [&](auto... levels) { return simdrng::XoshiroSIMD{seed}.stream(levels...); };
  stream_level_contract::check_stream_levels<simdrng::XoshiroSIMD>(make_simd);
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  const auto make_native = [&](auto... levels) { return simdrng::XoshiroNative{seed}.stream(levels...); };
  stream_level_contract::check_stream_levels<simdrng::XoshiroNative>(make_native);
#endif
}

TEST_CASE("TAG CONSTRUCTOR MATCHES STREAM", "[xoshiro256++][stream]") {
  const std::uint64_t seed = 42;
  const simdrng::XoshiroSIMD by_ctor{seed, simdrng::node{1}, simdrng::gpu{2}, simdrng::thread{3}};
  const auto by_stream = simdrng::XoshiroSIMD{seed}.stream(simdrng::thread{3}, simdrng::gpu{2}, simdrng::node{1});
  std::array<std::uint64_t, 32> a{};
  std::array<std::uint64_t, 32> b{};
  by_ctor.get_flat_state(a.data());
  by_stream.get_flat_state(b.data());
  REQUIRE(a == b);
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  const simdrng::XoshiroNative n_ctor{seed, simdrng::node{1}, simdrng::gpu{2}, simdrng::thread{3}};
  const auto n_stream = simdrng::XoshiroNative{seed}.stream(simdrng::thread{3}, simdrng::gpu{2}, simdrng::node{1});
  a = {};
  b = {};
  n_ctor.get_flat_state(a.data());
  n_stream.get_flat_state(b.data());
  REQUIRE(a == b);
#endif
}

// Plain-seed output pinned to the pre-tag layout: lane i sits at i * 2^128.
// The constants are the observed states and first output of the plain
// constructor at HEAD, before the tag API landed.
TEST_CASE("PLAIN SEED PINNED", "[xoshiro256++][stream]") {
  constexpr std::uint64_t seed = 42;
  // XoshiroScalar{42} state and first output, and the state after one 2^128 jump.
  constexpr std::array<std::uint64_t, 4> lane0 = {0xbdd732262feb6e95ULL, 0x28efe333b266f103ULL, 0x47526757130f9f52ULL,
                                                  0x581ce1ff0e4ae394ULL};
  constexpr std::array<std::uint64_t, 4> lane1 = {0x81746704fde896b5ULL, 0x645e944932dae0aeULL, 0xf4776829231c282cULL,
                                                  0x2393f9798732dba1ULL};
  constexpr std::uint64_t first = 0xd0764d4f4476689fULL;
  simdrng::XoshiroSIMD simd{seed};
  std::array<std::uint64_t, 32> flat{};
  simd.get_flat_state(flat.data());
  const auto W = simd.simd_width();
  for (std::uint8_t r = 0; r < 4; ++r) {
    REQUIRE(flat[r * W] == lane0[r]);
    if (W > 1) {
      REQUIRE(flat[r * W + 1] == lane1[r]);
    }
  }
  REQUIRE(simd() == first);
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  simdrng::XoshiroNative native{seed};
  native.get_flat_state(flat.data());
  constexpr auto NW = simdrng::XoshiroNative::simd_width();
  for (std::uint8_t r = 0; r < 4; ++r) {
    REQUIRE(flat[r * NW] == lane0[r]);
    if (NW > 1) {
      REQUIRE(flat[r * NW + 1] == lane1[r]);
    }
  }
  REQUIRE(native() == first);
#endif
}

TEST_CASE("SIMD DISPATCH JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroSIMD simd(seed);
  simdrng::XoshiroSIMD simd2(seed);
  // Both should produce the same output
  for (int i = 0; i < 1024; ++i) {
    REQUIRE(simd() == simd2());
  }
  // After jump, they should diverge
  simd.jump();
  REQUIRE(simd() != simd2());
}

TEST_CASE("SIMD DISPATCH MID JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroSIMD simd(seed);
  simdrng::XoshiroSIMD simd2(seed);
  simd.mid_jump();
  REQUIRE(simd() != simd2());
}

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
constexpr auto SIMD_WIDTH = xsimd::simd_type<simdrng::XoshiroNative::result_type>::size;

TEST_CASE("SEED", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar reference(seed);
  simdrng::XoshiroNative rng(seed);
  REQUIRE(rng.getState(0) == reference.getState());
  for (auto i = 1UL; i < SIMD_WIDTH; ++i) {
    reference.jump();
    REQUIRE(rng.getState(i) == reference.getState());
  }
}

TEST_CASE("JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar reference(seed);
  simdrng::XoshiroNative rng(seed);
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    REQUIRE(rng.getState(i) == reference.getState());
    reference.jump();
  }
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    rng.jump();
  }
  for (auto i = 0U; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(rng.getState(i) == reference.getState());
    reference.jump();
  }
}

TEST_CASE("LONG JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar reference(seed);
  simdrng::XoshiroNative rng(seed);
  rng.long_jump();
  reference.long_jump();
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(rng.getState(i) == reference.getState());
    reference.jump();
  }
}

TEST_CASE("GENERATE UINT64", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroNative rng(seed);
  std::vector<simdrng::XoshiroScalar> reference;
  reference.reserve(SIMD_WIDTH);
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    reference.emplace_back(seed);
  }
  for (auto i = 1U; i < SIMD_WIDTH; ++i) {
    for (auto j = 0UL; j < i; ++j) {
      reference[i].jump();
    }
  }
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    REQUIRE(rng.getState(i) == reference[i].getState());
  }
  for (auto i = 0; i < tests; i += SIMD_WIDTH) {
    for (auto j = 0UL; j < SIMD_WIDTH; ++j) {
      INFO("i: " << i << " j: " << j);
      REQUIRE(rng() == reference[j]());
    }
  }
}

TEST_CASE("MID JUMP", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar reference(seed);
  simdrng::XoshiroNative rng(seed);
  rng.mid_jump();
  reference.mid_jump();
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(rng.getState(i) == reference.getState());
    reference.jump();
  }
}

TEST_CASE("JUMP_N PER LANE", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  // Lane i is seeded as the scalar generator after i jumps; jump(k) then
  // advances every lane by k. So lane i must match scalar(seed) + i jumps + jump(k).
  for (const std::uint64_t k :
       {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{5}, std::uint64_t{1000}, std::uint64_t{1} << 20}) {
    INFO("k: " << k);
    simdrng::XoshiroNative rng(seed);
    std::vector<simdrng::XoshiroScalar> reference;
    reference.reserve(SIMD_WIDTH);
    for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
      reference.emplace_back(seed);
      for (auto j = 0UL; j < i; ++j)
        reference[i].jump();
    }
    rng.jump(k);
    for (auto &ref : reference)
      ref.jump(k);
    for (auto i = 0U; i < SIMD_WIDTH; ++i) {
      INFO("i: " << i);
      REQUIRE(rng.getState(i) == reference[i].getState());
    }
  }
}

TEST_CASE("SIMD DISPATCH JUMP_N", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  const std::uint64_t k = (std::uint64_t{1} << 30) + 12345;
  simdrng::XoshiroSIMD dispatched(seed);
  simdrng::XoshiroNative native(seed);
  dispatched.jump(k);
  native.jump(k);
  for (auto i = 0U; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(dispatched() == native());
  }
}

TEST_CASE("JUMP_N POW2 PER LANE", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  // jump(pow2{128}) per lane must match the fixed jump() on the scalar reference.
  simdrng::XoshiroNative rng(seed);
  std::vector<simdrng::XoshiroScalar> reference;
  reference.reserve(SIMD_WIDTH);
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    reference.emplace_back(seed);
    for (auto j = 0UL; j < i; ++j)
      reference[i].jump();
  }
  rng.jump(simdrng::pow2{128});
  for (auto &ref : reference)
    ref.jump(); // jump() == 2^128 == pow2{128}
  for (auto i = 0U; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(rng.getState(i) == reference[i].getState());
  }
}

TEST_CASE("SIMD DISPATCH JUMP_N POW2", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroSIMD dispatched(seed);
  simdrng::XoshiroNative native(seed);
  dispatched.jump(simdrng::pow2{200});
  native.jump(simdrng::pow2{200});
  for (auto i = 0U; i < SIMD_WIDTH; ++i) {
    INFO("i: " << i);
    REQUIRE(dispatched() == native());
  }
}

TEST_CASE("GENERATE DOUBLE", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroNative rng(seed);
  std::vector<simdrng::XoshiroScalar> reference;
  reference.reserve(SIMD_WIDTH);
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    reference.emplace_back(seed);
  }
  for (auto i = 1U; i < SIMD_WIDTH; ++i) {
    for (auto j = 0UL; j < i; ++j) {
      reference[i].jump();
    }
  }
  for (auto i = 0UL; i < SIMD_WIDTH; ++i) {
    REQUIRE(rng.getState(i) == reference[i].getState());
  }
  for (auto i = 0; i < tests; i += SIMD_WIDTH) {
    for (auto j = 0UL; j < SIMD_WIDTH; ++j) {
      INFO("i: " << i << " j: " << j);
      REQUIRE(rng.uniform() == reference[j].uniform());
    }
  }
}
#endif // XSIMD_NO_SUPPORTED_ARCHITECTURE
