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
// One stream stride: 2^32 next() calls advance the Weyl state by 2^32 * gamma.
constexpr std::uint64_t kStreamStride = (std::uint64_t{1} << 32) * 0x9e3779b97f4a7c15ULL;

} // namespace

// stream() must be usable in constant expressions, node included. The node{0}
// identity also survives a dropped node fold; the fold value itself is pinned
// by the named runtime test below, so a mutation fails a test, not a build.
static_assert(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{2}).getState() == kStreamSeed + 2 * kStreamStride);
static_assert(simdrng::SplitMix{kStreamSeed}.stream(simdrng::node{0}).getState() == kStreamSeed);
static_assert(simdrng::SplitMix::min() == std::numeric_limits<std::uint64_t>::lowest());
static_assert(simdrng::SplitMix::max() == std::numeric_limits<std::uint64_t>::max());

TEST_CASE("splitmix stream thread{i} equals i stream steps", "[splitmix][stream]") {
  for (std::uint64_t i = 0; i <= 8; ++i) {
    INFO("i: " << i);
    simdrng::SplitMix base(kStreamSeed);
    base.setState(base.getState() + i * kStreamStride);
    REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i}).getState() == base.getState());
  }
}

TEST_CASE("splitmix stream gpu{v} equals v << 24 stream strides", "[splitmix][stream]") {
  // gpu sits in the high 8 index bits and one index unit is one 2^32 stride.
  // A thread value of v << 24 would exceed the 24-bit thread field, so the
  // identity is stated against the strided state instead.
  for (const std::uint64_t v : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{255}}) {
    INFO("v: " << v);
    simdrng::SplitMix strided(kStreamSeed);
    strided.setState(kStreamSeed + (v << 24) * kStreamStride);
    REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::gpu{v}).getState() == strided.getState());
  }
}

TEST_CASE("splitmix stream thread{i+1} equals thread{i} advanced 2^32 next() calls", "[splitmix][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{3}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    simdrng::SplitMix advanced = simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i});
    advanced.setState(advanced.getState() + kStreamStride);
    REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i + 1}).getState() == advanced.getState());
  }
}

TEST_CASE("splitmix stream composition at large thread level", "[splitmix][stream]") {
  const std::uint64_t i = std::uint64_t{1} << 20;
  simdrng::SplitMix advanced = simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i});
  advanced.setState(advanced.getState() + 3 * kStreamStride);
  REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i + 3}).getState() == advanced.getState());
}

TEST_CASE("splitmix adjacent thread streams share no value", "[splitmix][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    auto a = simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i});
    auto b = simdrng::SplitMix{kStreamSeed}.stream(simdrng::thread{i + 1});
    std::array<std::uint64_t, 64> oa{}, ob{};
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

TEST_CASE("splitmix node{n} mixes mix64(n) into the seed", "[splitmix][stream]") {
  // The same fold Philox and ChaCha apply to their key words: the seed state
  // xors mix64(n) before the index stride. mix64(1) = 0x910a2dec89025cc1
  // (pinned so a dropped node fold fails this test by name).
  constexpr std::uint64_t mix1 = 0x910a2dec89025cc1ULL;
  REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::node{1}).getState() == (kStreamSeed ^ mix1));
  REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::node{0}).getState() == kStreamSeed);
  REQUIRE(simdrng::SplitMix{kStreamSeed}.getState() == kStreamSeed);
  REQUIRE(simdrng::SplitMix{kStreamSeed}.stream(simdrng::node{1}, simdrng::thread{3}).getState() ==
          (kStreamSeed ^ mix1) + 3 * kStreamStride);
}

TEST_CASE("splitmix stream level contract", "[splitmix][stream]") {
  stream_level_contract::check_stream_levels<simdrng::SplitMix>(
      [](auto... levels) { return simdrng::SplitMix{kStreamSeed}.stream(levels...); });
}
