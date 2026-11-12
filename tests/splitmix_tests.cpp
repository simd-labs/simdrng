#include <array>
#include <catch2/catch_all.hpp>
#include <cstdint>
#include <limits>
#include <random>
#include <simdrng/splitmix.hpp>

#include "stream_levels_contract.hpp"

#include "splitmix64.c"

static constexpr auto tests = 1 << 15;

TEST_CASE("splitmix", "[splitmix]") {
  x = std::random_device{}();
  INFO("SEED: " << x);
  simdrng::SplitMix splitmix(x);
  for (int i = 0; i < tests; ++i) {
    REQUIRE(splitmix() == next());
  }
}

namespace {

constexpr std::uint64_t kStreamSeed = 0x0102030405060708ULL;
constexpr std::uint64_t kGamma = 0x9e3779b97f4a7c15ULL;
// detail::widths_tight: block 2^24 outputs (thread shift 24), gpu shift 48,
// node shift 56.
constexpr std::uint64_t kBlock = std::uint64_t{1} << 24;

// The plain generator advanced by exactly n outputs. The Weyl state moves by
// n * gamma, the identity the oracle test above pins to the reference.
simdrng::SplitMix advanced_by(simdrng::SplitMix g, std::uint64_t outputs) {
  g.setState(g.getState() + outputs * kGamma);
  return g;
}

} // namespace

// stream() must be usable in constant expressions, node included.
static_assert(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{2}).getState() ==
              kStreamSeed + (std::uint64_t{2} << 24) * kGamma);
static_assert(simdrng::SplitMix{kStreamSeed}.stream(simdrng::node{0}).getState() == kStreamSeed);
static_assert(simdrng::SplitMix::min() == std::numeric_limits<std::uint64_t>::lowest());
static_assert(simdrng::SplitMix::max() == std::numeric_limits<std::uint64_t>::max());

TEST_CASE("splitmix stream thread{v} equals v * 2^24 outputs advanced", "[splitmix][stream]") {
  for (const std::uint64_t v : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{0xFFFFFF}}) {
    INFO("v: " << v);
    REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{v}).getState() ==
            advanced_by(simdrng::SplitMix{kStreamSeed}, v * kBlock).getState());
  }
}

TEST_CASE("splitmix stream thread{1} equals 2^24 direct next() calls", "[splitmix][stream]") {
  // Direct stepping where cheap: one full block of Weyl steps.
  auto stepped = simdrng::SplitMix{kStreamSeed};
  for (std::uint64_t i = 0; i < kBlock; ++i) {
    stepped();
  }
  REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{1}).getState() == stepped.getState());
}

TEST_CASE("splitmix stream gpu{v} equals v * 2^48 outputs advanced", "[splitmix][stream]") {
  for (const std::uint64_t v : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{255}}) {
    INFO("v: " << v);
    REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::gpu{v}).getState() ==
            advanced_by(simdrng::SplitMix{kStreamSeed}, v * (std::uint64_t{1} << 48)).getState());
  }
}

TEST_CASE("splitmix stream node{v} equals v * 2^56 outputs advanced", "[splitmix][stream]") {
  for (const std::uint64_t v : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{255}}) {
    INFO("v: " << v);
    REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::node{v}).getState() ==
            advanced_by(simdrng::SplitMix{kStreamSeed}, v * (std::uint64_t{1} << 56)).getState());
  }
}

TEST_CASE("splitmix stream composition at large thread level", "[splitmix][stream]") {
  const std::uint64_t i = std::uint64_t{1} << 20;
  auto advanced = simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i});
  advanced = advanced_by(advanced, 3 * kBlock);
  REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i + 3}).getState() == advanced.getState());
}

TEST_CASE("splitmix adjacent thread streams tile at the block boundary", "[splitmix][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    const auto end_of_block = advanced_by(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i}), kBlock);
    REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i + 1}).getState() == end_of_block.getState());
  }
}

TEST_CASE("splitmix adjacent thread streams share no value", "[splitmix][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    const auto a = stream_level_contract::draw64(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i}));
    const auto b = stream_level_contract::draw64(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i + 1}));
    int shared = 0;
    for (const auto xv : a) {
      for (const auto yv : b) {
        shared += (xv == yv);
      }
    }
    REQUIRE(shared == 0);
  }
}

TEST_CASE("splitmix max in-range levels work and stay distinct", "[splitmix][stream]") {
  // node 8 bits (255), gpu 8 bits (255) and thread 24 bits (0xFFFFFF) are the
  // largest values that fit their fields.
  constexpr std::uint64_t max_distance =
      (std::uint64_t{255} << 56) | (std::uint64_t{255} << 48) | (std::uint64_t{0xFFFFFF} << 24);
  const auto maxed =
      simdrng::SplitMix{kStreamSeed}.stream(simdrng::node{255}, simdrng::gpu{255}, simdrng::thread{0xFFFFFFULL});
  REQUIRE(maxed.getState() == advanced_by(simdrng::SplitMix{kStreamSeed}, max_distance).getState());
  const auto below =
      simdrng::SplitMix{kStreamSeed}.stream(simdrng::node{255}, simdrng::gpu{255}, simdrng::thread{0xFFFFFEULL});
  REQUIRE(stream_level_contract::draw64(maxed) != stream_level_contract::draw64(below));
}

TEST_CASE("splitmix sol regression: folded-node alias pairs now differ", "[splitmix][stream]") {
  // Sol review pairs at seed 0: (node, gpu, thread) = (0x5e11dab8f22eba8b, 0,
  // 0) aliased (0, 0, 1), and node 0x61c8864680b583eb aliased node 0. The raw
  // node values are past the 8-bit node field and stop the program there. The
  // regression maps them into the field (mod 2^8) and requires distinct
  // streams.
  constexpr std::uint64_t kPairNode = 0x5e11dab8f22eba8bULL & 0xFF;
  constexpr std::uint64_t kInvZeroNode = 0x61c8864680b583ebULL & 0xFF;
  REQUIRE(stream_level_contract::draw64(simdrng::SplitMix{0}.stream(simdrng::node{kPairNode})) !=
          stream_level_contract::draw64(simdrng::SplitMix{0}.stream(simdrng::thread{1})));
  REQUIRE(stream_level_contract::draw64(simdrng::SplitMix{0}.stream(simdrng::node{kInvZeroNode})) !=
          stream_level_contract::draw64(simdrng::SplitMix{0}.stream(simdrng::node{0})));
}

TEST_CASE("splitmix stream level contract", "[splitmix][stream]") {
  stream_level_contract::check_stream_levels<simdrng::SplitMix>(
      [](auto... levels) { return simdrng::SplitMix{kStreamSeed}.stream(levels...); });
}
