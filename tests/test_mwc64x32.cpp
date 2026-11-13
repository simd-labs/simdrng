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

namespace {

constexpr std::uint64_t kStreamSeed = 0x0102030405060708ULL;
// detail::widths_mwc_tight: block 2^22 outputs (thread shift 22), gpu shift 46,
// node shift 54.
constexpr std::uint64_t kBlock = std::uint64_t{1} << 22;

// The period of the generator is (MOD - 1) / 2 with MOD = 0xfff145f7ffffffff.
static_assert((0xfff145f7ffffffffULL - 1) / 2 == 0x7ff8a2fbffffffffULL);
// The table block length must keep the full address space inside the period:
// 2^(8+8+24+22) = 2^62 <= period, while 2^(8+8+24+23) = 2^63 > period.
static_assert(simdrng::detail::widths_mwc_tight.block_bits == 22);
static_assert((std::uint64_t{1} << (8 + 8 + 24 + 22)) <= 0x7ff8a2fbffffffffULL);

// The plain generator advanced by exactly n outputs, via the generator's own
// jump machinery (jump(n) is exact, the C reference jump_n pins it above).
simdrng::MWC64X32 advanced_by(simdrng::MWC64X32 g, std::uint64_t outputs) {
  g.jump(outputs);
  return g;
}

} // namespace

TEST_CASE("mwc64x32 stream thread{v} equals jump(v * 2^22)", "[mwc64x32][stream]") {
  for (const std::uint64_t seed : kSeeds) {
    for (const std::uint64_t v : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{0xFFFFFF}}) {
      INFO("seed: " << seed << " v: " << v);
      REQUIRE(simdrng::MWC64X32{seed}.stream(simdrng::thread{v}).getState() ==
              advanced_by(simdrng::MWC64X32{seed}, v * kBlock).getState());
    }
  }
}

TEST_CASE("mwc64x32 stream gpu{v} equals jump(v * 2^46)", "[mwc64x32][stream]") {
  for (const std::uint64_t seed : kSeeds) {
    for (const std::uint64_t v : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{255}}) {
      INFO("seed: " << seed << " v: " << v);
      REQUIRE(simdrng::MWC64X32{seed}.stream(simdrng::gpu{v}).getState() ==
              advanced_by(simdrng::MWC64X32{seed}, v * (std::uint64_t{1} << 46)).getState());
    }
  }
}

TEST_CASE("mwc64x32 stream node{v} equals jump(v * 2^54)", "[mwc64x32][stream]") {
  for (const std::uint64_t seed : kSeeds) {
    for (const std::uint64_t v : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{255}}) {
      INFO("seed: " << seed << " v: " << v);
      REQUIRE(simdrng::MWC64X32{seed}.stream(simdrng::node{v}).getState() ==
              advanced_by(simdrng::MWC64X32{seed}, v * (std::uint64_t{1} << 54)).getState());
    }
  }
}

TEST_CASE("mwc64x32 stream composition at large thread level", "[mwc64x32][stream]") {
  // Composing two smaller jumps: thread{i + 3} is thread{i} plus 3 blocks.
  const std::uint64_t i = std::uint64_t{1} << 20;
  const auto advanced = advanced_by(simdrng::MWC64X32{kSeeds[0]}.stream(simdrng::thread{i}), 3 * kBlock);
  REQUIRE(simdrng::MWC64X32{kSeeds[0]}.stream(simdrng::thread{i + 3}).getState() == advanced.getState());
}

TEST_CASE("mwc64x32 adjacent thread streams tile at the block boundary", "[mwc64x32][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    const auto end_of_block = advanced_by(simdrng::MWC64X32{kSeeds[0]}.stream(simdrng::thread{i}), kBlock);
    REQUIRE(simdrng::MWC64X32{kSeeds[0]}.stream(simdrng::thread{i + 1}).getState() == end_of_block.getState());
  }
}

TEST_CASE("mwc64x32 adjacent thread streams share no value", "[mwc64x32][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    const auto a = stream_level_contract::draw64(simdrng::MWC64X32{kSeeds[0]}.stream(simdrng::thread{i}));
    const auto b = stream_level_contract::draw64(simdrng::MWC64X32{kSeeds[0]}.stream(simdrng::thread{i + 1}));
    int shared = 0;
    for (const auto xv : a) {
      for (const auto yv : b) {
        shared += (xv == yv);
      }
    }
    REQUIRE(shared == 0);
  }
}

TEST_CASE("mwc64x32 max in-range levels work and stay distinct", "[mwc64x32][stream]") {
  // node 8 bits (255), gpu 8 bits (255) and thread 24 bits (0xFFFFFF) are the
  // largest values that fit their fields.
  constexpr std::uint64_t max_distance =
      (std::uint64_t{255} << 54) | (std::uint64_t{255} << 46) | (std::uint64_t{0xFFFFFF} << 22);
  const auto maxed =
      simdrng::MWC64X32{kStreamSeed}.stream(simdrng::node{255}, simdrng::gpu{255}, simdrng::thread{0xFFFFFFULL});
  REQUIRE(maxed.getState() == advanced_by(simdrng::MWC64X32{kStreamSeed}, max_distance).getState());
  const auto below =
      simdrng::MWC64X32{kStreamSeed}.stream(simdrng::node{255}, simdrng::gpu{255}, simdrng::thread{0xFFFFFEULL});
  REQUIRE(stream_level_contract::draw64(maxed) != stream_level_contract::draw64(below));
}

TEST_CASE("mwc64x32 sol regression: fallback and reduction node pairs now differ", "[mwc64x32][stream]") {
  // Sol review pairs at seed 42: node 0xd63e08b5413d386f restored the node-zero
  // state, and nodes 0x259daed25dd75561 and 0x3b55759a160e1082 reduced to one
  // state. The raw node values are past the 8-bit node field and stop the
  // program there. The regression maps them into the field (mod 2^8) and
  // requires distinct streams.
  constexpr std::uint64_t kFallbackNode = 0xd63e08b5413d386fULL & 0xFF;
  constexpr std::uint64_t kReductionA = 0x259daed25dd75561ULL & 0xFF;
  constexpr std::uint64_t kReductionB = 0x3b55759a160e1082ULL & 0xFF;
  REQUIRE(stream_level_contract::draw64(simdrng::MWC64X32{42}.stream(simdrng::node{kFallbackNode})) !=
          stream_level_contract::draw64(simdrng::MWC64X32{42}.stream(simdrng::node{0})));
  REQUIRE(stream_level_contract::draw64(simdrng::MWC64X32{42}.stream(simdrng::node{kReductionA})) !=
          stream_level_contract::draw64(simdrng::MWC64X32{42}.stream(simdrng::node{kReductionB})));
}

TEST_CASE("mwc64x32 stream level contract", "[mwc64x32][stream]") {
  stream_level_contract::check_stream_levels<simdrng::MWC64X32>(
      [](auto... levels) { return simdrng::MWC64X32{kSeeds[0]}.stream(levels...); });
}

TEST_CASE("mwc64x32 sol regression: reviewed and max tuples share no state", "[mwc64x32][stream]") {
  // Sol review 2 found tuple (127, 248, 10681344) at distance 2^64 - 2^24 in
  // the old 2^24-block layout, one past the period 0x7ff8a2fbffffffff, so its
  // start equalled stream (0, 0, 0) after one output. With the 2^22 block the
  // full address space stays inside the period and no two tuples share a
  // state. Include the max-tuple versus tuple-zero pair.
  const auto tuple0 = simdrng::MWC64X32{42}.stream();
  const auto maxed = simdrng::MWC64X32{42}.stream(simdrng::node{255}, simdrng::gpu{255}, simdrng::thread{0xFFFFFFULL});
  const auto reviewed = simdrng::MWC64X32{42}.stream(simdrng::node{127}, simdrng::gpu{248}, simdrng::thread{10681344});
  auto offset1 = simdrng::MWC64X32{42};
  offset1();
  REQUIRE(reviewed.getState() != tuple0.getState());
  REQUIRE(reviewed.getState() != offset1.getState());
  REQUIRE(reviewed.getState() != maxed.getState());
  REQUIRE(maxed.getState() != tuple0.getState());
  REQUIRE(maxed.getState() != offset1.getState());
}

TEST_CASE("mwc64x32 zero tags equal omitted tags after a draw", "[mwc64x32][stream]") {
  stream_level_contract::check_zero_tags_after_draw(simdrng::MWC64X32{kStreamSeed}, {1, 2, 5});
}
