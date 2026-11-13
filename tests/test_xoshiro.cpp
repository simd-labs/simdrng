#include <catch2/catch_all.hpp>
#include <cstdint>
#include <limits>
#include <random>
#include <simdrng/xoshiro_scalar.hpp>

#include "stream_levels_contract.hpp"

#include "xoshiro256plusplus.c"

static constexpr auto tests = 1 << 15;

TEST_CASE("xoshiro256++", "[xoshiro256++]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar rng(seed);
  s[0] = rng.getState()[0];
  s[1] = rng.getState()[1];
  s[2] = rng.getState()[2];
  s[3] = rng.getState()[3];
  for (int i = 0; i < tests; ++i) {
    REQUIRE(rng() == next());
  }
  rng.jump();
  jump();
  REQUIRE(rng.getState()[0] == s[0]);
  REQUIRE(rng.getState()[1] == s[1]);
  REQUIRE(rng.getState()[2] == s[2]);
  REQUIRE(rng.getState()[3] == s[3]);
  for (int i = 0; i < tests; ++i) {
    const auto result = rng.uniform();
    REQUIRE(result >= 0);
    REQUIRE(result < 1);
  }
}

TEST_CASE("xoshiro256++ jump(n) forward oracle", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  // jump(k) must land on the same state as k raw next() calls.
  for (const std::uint64_t k : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3},
                                std::uint64_t{5}, std::uint64_t{1000}, std::uint64_t{1} << 20}) {
    INFO("k: " << k);
    simdrng::XoshiroScalar jumped(seed);
    simdrng::XoshiroScalar forward(seed);
    jumped.jump(k);
    for (std::uint64_t i = 0; i < k; ++i)
      forward();
    REQUIRE(jumped.getState() == forward.getState());
  }
}

TEST_CASE("xoshiro256++ jump(n) composition", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<std::uint64_t> dist(0, (std::uint64_t{1} << 30) - 1);
  // jump(a); jump(b) == jump(a + b), for random a, b < 2^40 (exercises the
  // high bits of n in square-and-multiply that the forward oracle can't reach).
  for (int t = 0; t < 64; ++t) {
    const std::uint64_t a = dist(gen);
    const std::uint64_t b = dist(gen);
    INFO("a: " << a << " b: " << b);
    simdrng::XoshiroScalar split(seed);
    simdrng::XoshiroScalar combined(seed);
    split.jump(a);
    split.jump(b);
    combined.jump(a + b);
    REQUIRE(split.getState() == combined.getState());
  }
}

TEST_CASE("xoshiro256++ jump(0) is identity", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  simdrng::XoshiroScalar rng(seed);
  const auto before = rng.getState();
  rng.jump(0);
  REQUIRE(rng.getState() == before);
}

TEST_CASE("xoshiro256++ jump(pow2) matches step count", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  // For e < 64 the 2^e stride also fits the uint64 step count, so the two
  // independent poly paths (jump_poly_pow2 vs jump_poly_n) must agree.
  for (const std::uint64_t e : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{5}, std::uint64_t{20},
                                std::uint64_t{40}, std::uint64_t{63}}) {
    INFO("e: " << e);
    simdrng::XoshiroScalar by_pow2(seed);
    simdrng::XoshiroScalar by_steps(seed);
    by_pow2.jump(simdrng::pow2{e});
    by_steps.jump(std::uint64_t{1} << e);
    REQUIRE(by_pow2.getState() == by_steps.getState());
  }
}

TEST_CASE("xoshiro256++ jump(pow2) matches fixed jumps", "[xoshiro256++][jump_n]") {
  const auto seed = std::random_device()();
  INFO("SEED: " << seed);
  // The 2^128/2^160/2^192 strides exceed uint64, so validate jump_poly_pow2 at
  // those exponents against the trusted hardcoded jump constants.
  {
    simdrng::XoshiroScalar a(seed), b(seed);
    a.jump(simdrng::pow2{128});
    b.jump();
    REQUIRE(a.getState() == b.getState());
  }
  {
    simdrng::XoshiroScalar a(seed), b(seed);
    a.jump(simdrng::pow2{160});
    b.mid_jump();
    REQUIRE(a.getState() == b.getState());
  }
  {
    simdrng::XoshiroScalar a(seed), b(seed);
    a.jump(simdrng::pow2{192});
    b.long_jump();
    REQUIRE(a.getState() == b.getState());
  }
}

namespace {

constexpr std::uint64_t kStreamSeed = 0x0102030405060708ULL;
// detail::widths_xoshiro256: block 2^127 outputs (thread shift 127), gpu
// shift 159, node shift 191. The strides line up with the fixed jumps:
// thread{2} = 2^128 = jump(), gpu{2} = 2^160 = mid_jump(), node{2} =
// 2^192 = long_jump().
constexpr std::uint64_t kBlockExp = 127;

// The plain generator advanced by exactly 2^kBlockExp outputs.
simdrng::XoshiroScalar advanced_one_block(simdrng::XoshiroScalar g) {
  g.jump(simdrng::pow2{kBlockExp});
  return g;
}

} // namespace

// stream() must be usable in constant expressions. The tagged case runs at
// runtime instead: its jump_poly_pow2 loop exceeds the constexpr step limit
// of clang and MSVC (1048576) at compile time.
static_assert(simdrng::XoshiroScalar{kStreamSeed}.stream().getState() ==
              simdrng::XoshiroScalar(kStreamSeed).getState());
static_assert(simdrng::XoshiroScalar::min() == std::numeric_limits<simdrng::XoshiroScalar::result_type>::min());
static_assert(simdrng::XoshiroScalar::max() == std::numeric_limits<simdrng::XoshiroScalar::result_type>::max());

TEST_CASE("xoshiro256++ constexpr stream check at runtime", "[xoshiro256++][stream]") {
  REQUIRE(simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::thread{3})() !=
          simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::thread{4})());
}

TEST_CASE("xoshiro256++ stream thread{2} equals jump()", "[xoshiro256++][stream]") {
  simdrng::XoshiroScalar jumped(kStreamSeed);
  jumped.jump();
  REQUIRE(simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::thread{2}).getState() == jumped.getState());
}

TEST_CASE("xoshiro256++ stream gpu{2} equals mid_jump()", "[xoshiro256++][stream]") {
  simdrng::XoshiroScalar jumped(kStreamSeed);
  jumped.mid_jump();
  REQUIRE(simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::gpu{2}).getState() == jumped.getState());
}

TEST_CASE("xoshiro256++ stream node{2} equals long_jump()", "[xoshiro256++][stream]") {
  simdrng::XoshiroScalar jumped(kStreamSeed);
  jumped.long_jump();
  REQUIRE(simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::node{2}).getState() == jumped.getState());
}

TEST_CASE("xoshiro256++ stream thread{v} equals v * 2^127 jumps", "[xoshiro256++][stream]") {
  for (std::uint64_t v = 0; v <= 4; ++v) {
    INFO("v: " << v);
    auto by_jumps = simdrng::XoshiroScalar{kStreamSeed};
    for (std::uint64_t j = 0; j < v; ++j) {
      by_jumps = advanced_one_block(by_jumps);
    }
    REQUIRE(simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::thread{v}).getState() == by_jumps.getState());
  }
}

TEST_CASE("xoshiro256++ stream composition at large thread level", "[xoshiro256++][stream]") {
  // Composing two smaller jumps: thread{i + 3} is thread{i} plus 3 blocks.
  const std::uint64_t i = std::uint64_t{1} << 30;
  auto advanced = simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::thread{i});
  for (int j = 0; j < 3; ++j) {
    advanced = advanced_one_block(advanced);
  }
  REQUIRE(simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::thread{i + 3}).getState() == advanced.getState());
}

TEST_CASE("xoshiro256++ adjacent thread streams tile at the block boundary", "[xoshiro256++][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 30}) {
    INFO("i: " << i);
    const auto end_of_block = advanced_one_block(simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::thread{i}));
    REQUIRE(simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::thread{i + 1}).getState() == end_of_block.getState());
  }
}

TEST_CASE("xoshiro256++ adjacent thread streams share no value", "[xoshiro256++][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 30}) {
    INFO("i: " << i);
    const auto a = stream_level_contract::draw64(simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::thread{i}));
    const auto b = stream_level_contract::draw64(simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::thread{i + 1}));
    int shared = 0;
    for (const auto xv : a) {
      for (const auto yv : b) {
        shared += (xv == yv);
      }
    }
    REQUIRE(shared == 0);
  }
}

TEST_CASE("xoshiro256++ max in-range levels work and stay distinct", "[xoshiro256++][stream]") {
  // gpu and thread hold 32 bits (0xFFFFFFFF), the largest values that fit.
  // The max tuple equals the max-1 tuple plus one block, and the two draw
  // sets differ.
  const auto maxed =
      simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::gpu{0xFFFFFFFFULL}, simdrng::thread{0xFFFFFFFFULL});
  const auto below =
      simdrng::XoshiroScalar{kStreamSeed}.stream(simdrng::gpu{0xFFFFFFFFULL}, simdrng::thread{0xFFFFFFFEULL});
  REQUIRE(advanced_one_block(below).getState() == maxed.getState());
  REQUIRE(stream_level_contract::draw64(maxed) != stream_level_contract::draw64(below));
}

TEST_CASE("xoshiro256++ sol regression: seed 42 node{42} state is not zero", "[xoshiro256++][stream]") {
  // Sol review: a node equal to the seed cancelled the whole state, and every
  // thread then drew zeros.
  const auto g = simdrng::XoshiroScalar{42}.stream(simdrng::node{42});
  const auto st = g.getState();
  REQUIRE_FALSE((st[0] == 0 && st[1] == 0 && st[2] == 0 && st[3] == 0));
  REQUIRE(stream_level_contract::draw64(simdrng::XoshiroScalar{42}.stream(simdrng::node{42}, simdrng::thread{1})) !=
          stream_level_contract::draw64(simdrng::XoshiroScalar{42}.stream(simdrng::node{42}, simdrng::thread{2})));
}

TEST_CASE("xoshiro256++ stream level contract", "[xoshiro256++][stream]") {
  stream_level_contract::check_stream_levels<simdrng::XoshiroScalar>(
      [](auto... levels) { return simdrng::XoshiroScalar{kStreamSeed}.stream(levels...); });
}

TEST_CASE("xoshiro256++ zero tags equal omitted tags after a draw", "[xoshiro256++][stream]") {
  stream_level_contract::check_zero_tags_after_draw(simdrng::XoshiroScalar{kStreamSeed}, {1, 2, 5});
}
