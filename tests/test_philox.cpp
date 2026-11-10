#include <random>

#include <catch2/catch_all.hpp>

#include <simdrng/philox.hpp>

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

// The stream index fills the top 64 counter bits (words N-2..N-1 for W=32);
// the low words are the per-thread position.
simdrng::Philox4x32::counter_type expected_counter(std::uint64_t i) {
  simdrng::Philox4x32::counter_type ctr{};
  ctr[2] = static_cast<std::uint32_t>(i);
  ctr[3] = static_cast<std::uint32_t>(i >> 32);
  return ctr;
}

// The stream layout the stream tests check, for every Philox variant: the
// index takes the top 64 counter bits (the top 32 for Philox2x32).
template <class G> auto expected_stream_counter(std::uint64_t i) {
  using counter_t = typename G::counter_type;
  using word_t = typename G::word_type;
  constexpr std::size_t n = counter_t{}.size();
  counter_t ctr{};
  if constexpr (n == 2 && sizeof(word_t) == 4) {
    // Philox2x32: the index fills the top 32-bit word.
    ctr[1] = static_cast<word_t>(i);
  } else if (n == 4 && sizeof(word_t) == 4) {
    // Philox4x32: the index fills the top 64 counter bits.
    ctr[2] = static_cast<word_t>(i);
    ctr[3] = static_cast<word_t>(i >> 32);
  } else {
    // Philox2x64 / Philox4x64: the index fills the top 64-bit word.
    ctr[n - 1] = static_cast<word_t>(i);
  }
  return ctr;
}

} // namespace

// stream() must be usable in constant expressions.
static_assert(simdrng::Philox4x32::stream(kStreamSeed, 3)() != simdrng::Philox4x32::stream(kStreamSeed, 4)());

TEST_CASE("Philox4x32 stream counter word holds the stream index", "[philox][stream]") {
  for (const std::uint64_t i :
       {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3}, std::uint64_t{4}, std::uint64_t{5},
        std::uint64_t{6}, std::uint64_t{7}, std::uint64_t{8}, std::uint64_t{1} << 40}) {
    INFO("i: " << i);
    REQUIRE(simdrng::Philox4x32::stream(kStreamSeed, i).getCounter() == expected_counter(i));
  }
}

TEST_CASE("Philox4x32 stream draws only advance the low counter words", "[philox][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{3}, std::uint64_t{1} << 40}) {
    INFO("i: " << i);
    auto s = simdrng::Philox4x32::stream(kStreamSeed, i);
    for (int k = 0; k < 16; ++k) {
      s();
    }
    // 16 outputs = 8 blocks: position 8 in the low word, index words unchanged.
    auto expected = expected_counter(i);
    expected[0] = 8;
    REQUIRE(s.getCounterForSerde() == expected);
  }
}

TEST_CASE("Philox4x32 adjacent streams share no value", "[philox][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 40}) {
    INFO("i: " << i);
    auto a = simdrng::Philox4x32::stream(kStreamSeed, i);
    auto b = simdrng::Philox4x32::stream(kStreamSeed, i + 1);
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

TEMPLATE_TEST_CASE("Philox stream counter holds the index in the top counter half", "[philox][stream]",
                   simdrng::Philox4x32, simdrng::Philox2x32, simdrng::Philox4x64, simdrng::Philox2x64) {
  using G = TestType;
  // counter bits / 2 = index bits, capped at the 64-bit stream index.
  constexpr unsigned index_bits = std::min<unsigned>(static_cast<unsigned>(sizeof(typename G::counter_type) * 4), 64u);
  const std::uint64_t max_index = (index_bits >= 64) ? ~std::uint64_t{0} : (std::uint64_t{1} << index_bits) - 1;
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 31, max_index}) {
    INFO("i: " << i);
    REQUIRE(G::stream(kStreamSeed, i).getCounter() == expected_stream_counter<G>(i));
  }
}

TEMPLATE_TEST_CASE("Philox streams 0 and 1 share no value in their first 64 draws", "[philox][stream]",
                   simdrng::Philox4x32, simdrng::Philox2x32, simdrng::Philox4x64, simdrng::Philox2x64) {
  using G = TestType;
  auto a = G::stream(kStreamSeed, 0);
  auto b = G::stream(kStreamSeed, 1);
  std::array<std::uint64_t, 64> oa{}, ob{};
  for (auto &v : oa) {
    v = a();
  }
  for (auto &v : ob) {
    v = b();
  }
  int shared = 0;
  for (const auto x : oa) {
    for (const auto y : ob) {
      shared += (x == y);
    }
  }
  REQUIRE(shared == 0);
}

TEST_CASE("Philox2x32 stream 0 second draw differs from stream 1 first draw", "[philox][stream]") {
  using G = simdrng::Philox2x32;
  auto s0 = G::stream(kStreamSeed, 0);
  s0();
  const auto s0_second = s0();
  const auto s1_first = G::stream(kStreamSeed, 1)();
  REQUIRE(s0_second != s1_first);
}

TEST_CASE("Philox2x32 stream index wraps at 2^32", "[philox][stream]") {
  // The index half is 32 bits, like SplitMix::stream: i >= 2^32 repeats a stream.
  using G = simdrng::Philox2x32;
  REQUIRE(G::stream(kStreamSeed, std::uint64_t{1} << 32).getCounter() == G::stream(kStreamSeed, 0).getCounter());
  REQUIRE(G::stream(kStreamSeed, (std::uint64_t{1} << 32) + 5).getCounter() == G::stream(kStreamSeed, 5).getCounter());
}
