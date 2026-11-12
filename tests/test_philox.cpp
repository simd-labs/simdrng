#include <random>

#include <catch2/catch_all.hpp>

#include <simdrng/philox.hpp>

#include "stream_levels_contract.hpp"

// Random123 Known Answer Test (KAT) vectors for Philox4x32-10
// Source: Random123 reference implementation (philox.h, kat_vectors)
TEST_CASE("Philox4x32-10 reference vectors", "[philox]") {
  using P = simdrng::Philox<4, 32, 10>;
  using ctr_t = P::counter_type;
  using key_type = P::key_type;

  auto check_block = [](key_type key, ctr_t ctr, ctr_t expected) {
    P rng(key, ctr);
    rng();
    auto output = std::bit_cast<ctr_t>(rng.result_cache());
    REQUIRE(output[0] == expected[0]);
    REQUIRE(output[1] == expected[1]);
    REQUIRE(output[2] == expected[2]);
    REQUIRE(output[3] == expected[3]);
  };

  SECTION("zeros") { check_block({0, 0}, {0, 0, 0, 0}, {0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8}); }

  SECTION("all ones") {
    check_block({0xffffffff, 0xffffffff}, {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff},
                {0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd});
  }

  SECTION("pi digits") {
    check_block({0xa4093822, 0x299f31d0}, {0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344},
                {0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1});
  }
}

// Random123 KAT vectors for Philox2x32-10
TEST_CASE("Philox2x32-10 reference vectors", "[philox]") {
  using P = simdrng::Philox<2, 32, 10>;
  using ctr_t = P::counter_type;
  using key_type = P::key_type;

  auto check_block = [](key_type key, ctr_t ctr, ctr_t expected) {
    P rng(key, ctr);
    rng();
    auto output = std::bit_cast<ctr_t>(rng.result_cache());
    REQUIRE(output[0] == expected[0]);
    REQUIRE(output[1] == expected[1]);
  };

  SECTION("zeros") { check_block({0}, {0, 0}, {0xff1dae59, 0x6cd10df2}); }

  SECTION("all ones") { check_block({0xffffffff}, {0xffffffff, 0xffffffff}, {0x2c3f628b, 0xab4fd7ad}); }

  SECTION("pi digits") { check_block({0x13198a2e}, {0x243f6a88, 0x85a308d3}, {0xdd7ce038, 0xf62a4c12}); }
}

// Random123 KAT vectors for Philox4x64-10
TEST_CASE("Philox4x64-10 reference vectors", "[philox]") {
  using P = simdrng::Philox<4, 64, 10>;
  using ctr_t = P::counter_type;
  using key_type = P::key_type;

  SECTION("zeros") {
    P rng(key_type{0, 0}, ctr_t{0, 0, 0, 0});
    REQUIRE(rng() == 0x16554d9eca36314cULL);
    REQUIRE(rng() == 0xdb20fe9d672d0fdcULL);
    REQUIRE(rng() == 0xd7e772cee186176bULL);
    REQUIRE(rng() == 0x7e68b68aec7ba23bULL);
  }

  SECTION("all ones") {
    P rng(key_type{0xffffffffffffffffULL, 0xffffffffffffffffULL},
          ctr_t{0xffffffffffffffffULL, 0xffffffffffffffffULL, 0xffffffffffffffffULL, 0xffffffffffffffffULL});
    REQUIRE(rng() == 0x87b092c3013fe90bULL);
    REQUIRE(rng() == 0x438c3c67be8d0224ULL);
    REQUIRE(rng() == 0x9cc7d7c69cd777b6ULL);
    REQUIRE(rng() == 0xa09caebf594f0ba0ULL);
  }
}

// Random123 KAT vectors for Philox2x64-10
TEST_CASE("Philox2x64-10 reference vectors", "[philox]") {
  using P = simdrng::Philox<2, 64, 10>;
  using ctr_t = P::counter_type;
  using key_type = P::key_type;

  SECTION("zeros") {
    P rng(key_type{0}, ctr_t{0, 0});
    REQUIRE(rng() == 0xca00a0459843d731ULL);
    REQUIRE(rng() == 0x66c24222c9a845b5ULL);
  }

  SECTION("all ones") {
    P rng(key_type{0xffffffffffffffffULL}, ctr_t{0xffffffffffffffffULL, 0xffffffffffffffffULL});
    REQUIRE(rng() == 0x65b021d60cd8310fULL);
    REQUIRE(rng() == 0x4d02f3222f86df20ULL);
  }
}

// Verify sequential output: calling operator() returns elements from the same block
TEST_CASE("Philox4x32 sequential output matches block", "[philox]") {
  simdrng::Philox4x32 rng({0x12345678, 0x9abcdef0}, {0, 0, 0, 0});
  auto r0 = rng();
  auto r1 = rng();
  simdrng::Philox4x32 rng2({0x12345678, 0x9abcdef0}, {0, 0, 0, 0});
  rng2();
  auto cache = rng2.result_cache();
  REQUIRE(r0 == cache[0]);
  REQUIRE(r1 == cache[1]);
}

// Verify counter increment produces different output
TEST_CASE("Philox4x32 different counters produce different output", "[philox]") {
  simdrng::Philox4x32 rng1({0, 0}, {0, 0, 0, 0});
  simdrng::Philox4x32 rng2({0, 0}, {1, 0, 0, 0});
  REQUIRE(rng1() != rng2());
}

// Verify seed constructor produces deterministic output
TEST_CASE("Philox seed constructor is deterministic", "[philox]") {
  simdrng::Philox4x32 a(42);
  simdrng::Philox4x32 b(42);
  for (int i = 0; i < 100; ++i) {
    REQUIRE(a() == b());
  }
}

// Verify long sequence consistency across all four variants
TEMPLATE_TEST_CASE("Philox deterministic sequence", "[philox]", simdrng::Philox4x32, simdrng::Philox2x32,
                   simdrng::Philox4x64, simdrng::Philox2x64) {
  auto seed = std::random_device{}();
  INFO("SEED: " << seed);
  TestType a(static_cast<uint64_t>(seed));
  TestType b(static_cast<uint64_t>(seed));
  for (int i = 0; i < 10000; ++i) {
    REQUIRE(a() == b());
  }
}

namespace {

constexpr std::uint64_t kStreamSeed = 0x0102030405060708ULL;

// The literal counter block for a level tuple: node * 2^a + gpu * 2^b +
// thread * 2^c counter blocks, spelled per variant from the width table in
// detail::stream_widths (64-bit counter: shifts 56/48/24, block 2^24;
// 128-bit counters: shifts 112/96/64, block 2^64; 256-bit counter: shifts
// 192/128/64, block 2^64). The low words are the position inside the block.
template <class G> auto expected_stream_counter(std::uint64_t node_v, std::uint64_t gpu_v, std::uint64_t thread_v) {
  using counter_t = typename G::counter_type;
  using word_t = typename G::word_type;
  counter_t ctr{};
  if constexpr (counter_t{}.size() == 2 && sizeof(word_t) == 4) {
    // Philox2x32: distance = node * 2^56 + gpu * 2^48 + thread * 2^24.
    ctr[0] = static_cast<word_t>(thread_v << 24);
    ctr[1] = static_cast<word_t>((thread_v >> 8) | (gpu_v << 16) | (node_v << 24));
  } else if constexpr (counter_t{}.size() == 4 && sizeof(word_t) == 4) {
    // Philox4x32: distance = node * 2^112 + gpu * 2^96 + thread * 2^64.
    ctr[2] = static_cast<word_t>(thread_v);
    ctr[3] = static_cast<word_t>(gpu_v | (node_v << 16));
  } else if constexpr (counter_t{}.size() == 2) {
    // Philox2x64: distance = node * 2^112 + gpu * 2^96 + thread * 2^64.
    ctr[1] = static_cast<word_t>(thread_v | (gpu_v << 32) | (node_v << 48));
  } else {
    // Philox4x64: distance = node * 2^192 + gpu * 2^128 + thread * 2^64.
    ctr[1] = static_cast<word_t>(thread_v);
    ctr[2] = static_cast<word_t>(gpu_v);
    ctr[3] = static_cast<word_t>(node_v);
  }
  return ctr;
}

// The largest value that fits the node/gpu fields of one variant.
template <class G> constexpr std::uint64_t max_node() {
  using counter_t = typename G::counter_type;
  if constexpr (counter_t{}.size() == 2 && sizeof(typename G::word_type) == 4) {
    return 255;
  } else if constexpr (counter_t{}.size() == 4 && sizeof(typename G::word_type) == 4) {
    return 0xFFFF;
  } else if constexpr (counter_t{}.size() == 2) {
    return 0xFFFF;
  } else {
    return 0xFFFFFFFFFFFFFFFFULL;
  }
}

// The largest value that fits the thread field of one variant.
template <class G> constexpr std::uint64_t max_thread() {
  using counter_t = typename G::counter_type;
  if constexpr (counter_t{}.size() == 2 && sizeof(typename G::word_type) == 4) {
    return 0xFFFFFF;
  } else if constexpr (counter_t{}.size() == 4 && sizeof(typename G::word_type) == 4) {
    return 0xFFFFFFFFULL;
  } else if constexpr (counter_t{}.size() == 2) {
    return 0xFFFFFFFFULL;
  } else {
    return 0xFFFFFFFFFFFFFFFFULL;
  }
}

} // namespace

// stream() must be usable in constant expressions.
static_assert(simdrng::Philox4x32{kStreamSeed}.stream(simdrng::thread{3})() !=
              simdrng::Philox4x32{kStreamSeed}.stream(simdrng::thread{4})());

TEMPLATE_TEST_CASE("Philox stream counter holds the level tuple at the field shifts", "[philox][stream]",
                   simdrng::Philox4x32, simdrng::Philox2x32, simdrng::Philox4x64, simdrng::Philox2x64) {
  using G = TestType;
  const auto mn = max_node<G>();
  const auto mt = max_thread<G>();
  for (const std::uint64_t n : {std::uint64_t{0}, std::uint64_t{1}, mn}) {
    for (const std::uint64_t g : {std::uint64_t{0}, std::uint64_t{1}, mn}) {
      for (const std::uint64_t t : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 20, mt}) {
        INFO("node: " << n << " gpu: " << g << " thread: " << t);
        REQUIRE(G{kStreamSeed}.stream(simdrng::node{n}, simdrng::gpu{g}, simdrng::thread{t}).getCounter() ==
                expected_stream_counter<G>(n, g, t));
      }
    }
  }
}

TEMPLATE_TEST_CASE("Philox stream(level{v}) equals the plain generator advanced v * 2^field blocks", "[philox][stream]",
                   simdrng::Philox4x32, simdrng::Philox2x32, simdrng::Philox4x64, simdrng::Philox2x64) {
  using G = TestType;
  for (const std::uint64_t v : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{7}}) {
    INFO("v: " << v);
    const auto thread_ref = G{G::seed_to_key(kStreamSeed), expected_stream_counter<G>(0, 0, v)};
    REQUIRE(stream_level_contract::draw64(G{kStreamSeed}.stream(simdrng::thread{v})) ==
            stream_level_contract::draw64(thread_ref));
    const auto gpu_ref = G{G::seed_to_key(kStreamSeed), expected_stream_counter<G>(0, v, 0)};
    REQUIRE(stream_level_contract::draw64(G{kStreamSeed}.stream(simdrng::gpu{v})) ==
            stream_level_contract::draw64(gpu_ref));
    const auto node_ref = G{G::seed_to_key(kStreamSeed), expected_stream_counter<G>(v, 0, 0)};
    REQUIRE(stream_level_contract::draw64(G{kStreamSeed}.stream(simdrng::node{v})) ==
            stream_level_contract::draw64(node_ref));
  }
}

TEMPLATE_TEST_CASE("Philox one-level streams compose into the tuple stream", "[philox][stream]", simdrng::Philox4x32,
                   simdrng::Philox2x32, simdrng::Philox4x64, simdrng::Philox2x64) {
  using G = TestType;
  const auto composed = G{kStreamSeed}.stream(simdrng::node{5}).stream(simdrng::gpu{6}).stream(simdrng::thread{7});
  REQUIRE(composed.getCounter() ==
          G{kStreamSeed}.stream(simdrng::node{5}, simdrng::gpu{6}, simdrng::thread{7}).getCounter());
  REQUIRE(stream_level_contract::draw64(composed) ==
          stream_level_contract::draw64(G{kStreamSeed}.stream(simdrng::thread{7}, simdrng::node{5}, simdrng::gpu{6})));
}

TEST_CASE("Philox4x32 stream draws only advance the low counter words", "[philox][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{3}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    auto s = simdrng::Philox4x32{kStreamSeed}.stream(simdrng::thread{i});
    for (int k = 0; k < 16; ++k) {
      s();
    }
    // 16 outputs = 8 blocks: position 8 in the low word, level words unchanged.
    auto expected = expected_stream_counter<simdrng::Philox4x32>(0, 0, i);
    expected[0] = 8;
    REQUIRE(s.getCounterForSerde() == expected);
  }
}

TEMPLATE_TEST_CASE("Philox adjacent thread streams share no value", "[philox][stream]", simdrng::Philox4x32,
                   simdrng::Philox2x32, simdrng::Philox4x64, simdrng::Philox2x64) {
  using G = TestType;
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    const auto a = stream_level_contract::draw64(G{kStreamSeed}.stream(simdrng::thread{i}));
    const auto b = stream_level_contract::draw64(G{kStreamSeed}.stream(simdrng::thread{i + 1}));
    int shared = 0;
    for (const auto x : a) {
      for (const auto y : b) {
        shared += (x == y);
      }
    }
    REQUIRE(shared == 0);
  }
}

TEST_CASE("philox2x32 stream 0 second draw differs from stream 1 first draw", "[philox][stream]") {
  using G = simdrng::Philox2x32;
  const G base{kStreamSeed};
  auto s0 = base.stream();
  s0();
  const auto s0_second = s0();
  const auto s1_first = base.stream(simdrng::thread{1})();
  REQUIRE(s0_second != s1_first);
}

TEST_CASE("philox sol regression: folded-node alias pairs now differ", "[philox][stream]") {
  // Sol review pairs at seed 42: Philox2x32 nodes 0xf8364607e9c949bd and
  // 0x575d4575a1ecca9a produced one key, and node 0x61c8864680b583eb aliased
  // node 0 on Philox4x32 and Philox2x64. The raw node values are past the
  // node field (8 bits for Philox2x32, 16 bits for the 128-bit counters) and
  // stop the program there. The regression maps them into the field and
  // requires distinct streams.
  constexpr std::uint64_t kPairA = 0xf8364607e9c949bdULL & 0xFF;
  constexpr std::uint64_t kPairB = 0x575d4575a1ecca9aULL & 0xFF;
  constexpr std::uint64_t kInvZero = 0x61c8864680b583ebULL & 0xFFFF;
  REQUIRE(stream_level_contract::draw64(simdrng::Philox2x32{42}.stream(simdrng::node{kPairA})) !=
          stream_level_contract::draw64(simdrng::Philox2x32{42}.stream(simdrng::node{kPairB})));
  REQUIRE(stream_level_contract::draw64(simdrng::Philox4x32{42}.stream(simdrng::node{kInvZero})) !=
          stream_level_contract::draw64(simdrng::Philox4x32{42}.stream(simdrng::node{0})));
  REQUIRE(stream_level_contract::draw64(simdrng::Philox2x64{42}.stream(simdrng::node{kInvZero})) !=
          stream_level_contract::draw64(simdrng::Philox2x64{42}.stream(simdrng::node{0})));
}

TEMPLATE_TEST_CASE("Philox stream level contract", "[philox][stream]", simdrng::Philox4x32, simdrng::Philox2x32,
                   simdrng::Philox4x64, simdrng::Philox2x64) {
  using G = TestType;
  stream_level_contract::check_stream_levels<G>([](auto... levels) { return G{kStreamSeed}.stream(levels...); });
}
