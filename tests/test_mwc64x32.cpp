#include <array>
#include <catch2/catch_all.hpp>
#include <cstdint>
#include <limits>
#include <random>
#include <simdrng/mwc64x32.hpp>

#include "stream_levels_contract.hpp"

#if !defined(__SIZEOF_INT128__) || defined(SIMDRNG_TEST_U128_SHIM)
#include "mwc64x32_u128_shim.hpp"
#define __uint128_t mwc_ref::u128
#else
__extension__ typedef unsigned __int128 __uint128_t;
#endif

#include "MWC64X_32.c"

namespace {

// Vigna's seeding rule (see MWC64X_32.c): c = 1, x = the low 32 bits of the
// first SplitMix64 output from the seed.
void seed_reference(const std::uint64_t seed) {
  c = 1;
  x = static_cast<std::uint32_t>(simdrng::SplitMix{seed}());
}

constexpr std::array<std::uint64_t, 4> kSeeds = {0ULL, 1ULL, 42ULL, 0xDEADBEEFCAFEBABEULL};

} // namespace

TEST_CASE("mwc64x32 next matches the C reference", "[mwc64x32]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  seed_reference(seed);
  simdrng::MWC64X32 rng(seed);
  REQUIRE(rng.getState() == simdrng::array<std::uint32_t, 2>{x, c});
  for (int i = 0; i < (1 << 14); ++i) {
    REQUIRE(rng() == next());
  }
}

TEST_CASE("mwc64x32 seeded state and first outputs", "[mwc64x32]") {
  for (const std::uint64_t seed : kSeeds) {
    INFO("seed: " << seed);
    seed_reference(seed);
    simdrng::MWC64X32 rng(seed);
    REQUIRE(rng.getState() == simdrng::array<std::uint32_t, 2>{x, c});
    for (int i = 0; i < 16; ++i) {
      REQUIRE(rng() == next());
    }
  }
}

TEST_CASE("mwc64x32 jump and long_jump match the C reference", "[mwc64x32]") {
  for (const std::uint64_t seed : kSeeds) {
    INFO("seed: " << seed);
    seed_reference(seed);
    simdrng::MWC64X32 rng(seed);
    rng.jump();
    jump();
    REQUIRE(rng.getState() == simdrng::array<std::uint32_t, 2>{x, c});
    for (int i = 0; i < 4; ++i) {
      REQUIRE(rng() == next());
    }
    seed_reference(seed);
    rng = simdrng::MWC64X32(seed);
    rng.long_jump();
    long_jump();
    REQUIRE(rng.getState() == simdrng::array<std::uint32_t, 2>{x, c});
    for (int i = 0; i < 4; ++i) {
      REQUIRE(rng() == next());
    }
  }
}

TEST_CASE("mwc64x32 jump(n) matches the C reference jump_n", "[mwc64x32][jump_n]") {
  for (const std::uint64_t seed : kSeeds) {
    for (const std::uint64_t n : {0ULL, 1ULL, 2ULL, 3ULL, 5ULL, 1000ULL, 1ULL << 20, 1ULL << 32, 1ULL << 48}) {
      INFO("seed: " << seed << " n: " << n);
      seed_reference(seed);
      simdrng::MWC64X32 rng(seed);
      rng.jump(n);
      jump_n(n);
      REQUIRE(rng.getState() == simdrng::array<std::uint32_t, 2>{x, c});
    }
  }
}

TEST_CASE("mwc64x32 jump(1000) equals 1000 next() calls", "[mwc64x32][jump_n]") {
  for (const std::uint64_t seed : kSeeds) {
    INFO("seed: " << seed);
    simdrng::MWC64X32 jumped(seed);
    simdrng::MWC64X32 forward(seed);
    jumped.jump(1000ULL);
    for (int i = 0; i < 1000; ++i) {
      forward();
    }
    REQUIRE(jumped.getState() == forward.getState());
  }
}

TEST_CASE("mwc64x32 jump(2^32) equals jump()", "[mwc64x32][jump_n]") {
  for (const std::uint64_t seed : kSeeds) {
    INFO("seed: " << seed);
    simdrng::MWC64X32 by_steps(seed);
    simdrng::MWC64X32 fixed(seed);
    by_steps.jump(std::uint64_t{1} << 32);
    fixed.jump();
    REQUIRE(by_steps.getState() == fixed.getState());
  }
}

// stream() must be usable in constant expressions.
static_assert(simdrng::MWC64X32{42}.stream().getState() == simdrng::MWC64X32(42).getState());
static_assert(simdrng::MWC64X32::min() == std::numeric_limits<simdrng::MWC64X32::result_type>::min());
static_assert(simdrng::MWC64X32::max() == std::numeric_limits<simdrng::MWC64X32::result_type>::max());

TEST_CASE("mwc64x32 stream thread{i} equals i jumps", "[mwc64x32][stream]") {
  for (const std::uint64_t seed : kSeeds) {
    for (std::uint64_t i = 0; i <= 8; ++i) {
      INFO("seed: " << seed << " i: " << i);
      simdrng::MWC64X32 by_jumps(seed);
      for (std::uint64_t j = 0; j < i; ++j) {
        by_jumps.jump();
      }
      REQUIRE(simdrng::MWC64X32{seed}.stream(simdrng::thread{i}).getState() == by_jumps.getState());
    }
  }
}

TEST_CASE("mwc64x32 stream gpu{v} equals v jumps of 2^56 outputs", "[mwc64x32][stream]") {
  // gpu sits in the high 7 index bits: index v << 24, and one index unit
  // spans 2^32 outputs, so gpu{v} skips v * 2^56 outputs. A thread value of
  // v << 24 would exceed the 24-bit thread field, so the identity is stated
  // as a jump instead.
  for (const std::uint64_t seed : kSeeds) {
    INFO("seed: " << seed);
    for (const std::uint64_t v : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{127}}) {
      INFO("v: " << v);
      simdrng::MWC64X32 jumped(seed);
      jumped.jump(v << 56);
      REQUIRE(simdrng::MWC64X32{seed}.stream(simdrng::gpu{v}).getState() == jumped.getState());
    }
  }
}

TEST_CASE("mwc64x32 stream composition at large thread level", "[mwc64x32][stream]") {
  const std::uint64_t i = std::uint64_t{1} << 20;
  simdrng::MWC64X32 advanced = simdrng::MWC64X32{kSeeds[0]}.stream(simdrng::thread{i});
  advanced.jump();
  advanced.jump();
  advanced.jump();
  REQUIRE(simdrng::MWC64X32{kSeeds[0]}.stream(simdrng::thread{i + 3}).getState() == advanced.getState());
}

TEST_CASE("mwc64x32 adjacent thread streams share no value", "[mwc64x32][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    auto a = simdrng::MWC64X32{kSeeds[0]}.stream(simdrng::thread{i});
    auto b = simdrng::MWC64X32{kSeeds[0]}.stream(simdrng::thread{i + 1});
    std::array<std::uint32_t, 64> oa{}, ob{};
    for (auto &v : oa) {
      v = a();
    }
    for (auto &v : ob) {
      v = b();
    }
    int shared = 0;
    for (const auto xv : oa) {
      for (const auto yv : ob) {
        shared += (xv == yv);
      }
    }
    REQUIRE(shared == 0);
  }
}

TEST_CASE("mwc64x32 max in-range levels match pinned head output", "[mwc64x32][stream]") {
  // gpu 7 bits (127) and thread 24 bits (0xFFFFFF) are the largest values
  // that fit the 31-bit stream index. The draws are pinned to the output of
  // HEAD 16d018c at seed 0x0102030405060708; the range checks must not move
  // any in-range output.
  constexpr std::uint64_t kPinSeed = 0x0102030405060708ULL;
  constexpr std::uint32_t expected[] = {0x4b55e49cU, 0x52cc8e4dU, 0x589d0323U, 0x5f21797fU,
                                        0xa97797abU, 0x4e1f27a6U, 0x72397c26U, 0x94aeec91U};
  auto g = simdrng::MWC64X32{kPinSeed}.stream(simdrng::gpu{127}, simdrng::thread{0xFFFFFFULL});
  for (const auto want : expected) {
    REQUIRE(g() == want);
  }
}

TEST_CASE("mwc64x32 stream level contract", "[mwc64x32][stream]") {
  stream_level_contract::check_stream_levels<simdrng::MWC64X32>(
      [](auto... levels) { return simdrng::MWC64X32{kSeeds[0]}.stream(levels...); });
}
