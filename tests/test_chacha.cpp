#include <cstring>
#include <random>

#include <catch2/catch_all.hpp>
#include <monocypher.h>

#include <simdrng/chacha.hpp>

#include "stream_levels_contract.hpp"

static constexpr auto tests = 1 << 15;

TEST_CASE("ChaChaScalar", "[chacha]") {
  using ChaCha20 = simdrng::ChaCha<20>;

  auto seed = std::random_device{}();
  INFO("SEED: " << seed);
  std::mt19937 rng32(seed);
  std::mt19937_64 rng64(seed);
  ChaCha20::input_word counter = rng64(), nonce = rng64();
  std::array<ChaCha20::matrix_word, 8> key;
  for (int i = 0; i < 8; i++) {
    key[i] = rng32();
  }

  ChaCha20 rngChaCha(key, counter, nonce);
  for (int i = 0; i < tests; ++i) {
    // In a correct implementation, the internal state should neccsarily be
    // a validly arranged input for the algorithm and thus the reference impl.
    const auto input = rngChaCha.getState();
    uint8_t referenceOutput[64];

    // Assemble key (8 words -> 32 bytes) and nonce (2 words -> 8 bytes)
    uint8_t keyBytes[32];
    for (int k = 0; k < 8; ++k) {
      uint32_t w = input[4 + k];
      keyBytes[4 * k + 0] = static_cast<uint8_t>(w & 0xFF);
      keyBytes[4 * k + 1] = static_cast<uint8_t>((w >> 8) & 0xFF);
      keyBytes[4 * k + 2] = static_cast<uint8_t>((w >> 16) & 0xFF);
      keyBytes[4 * k + 3] = static_cast<uint8_t>((w >> 24) & 0xFF);
    }
    uint8_t nonceBytes[8];
    uint32_t n0 = input[14];
    uint32_t n1 = input[15];
    nonceBytes[0] = static_cast<uint8_t>(n0 & 0xFF);
    nonceBytes[1] = static_cast<uint8_t>((n0 >> 8) & 0xFF);
    nonceBytes[2] = static_cast<uint8_t>((n0 >> 16) & 0xFF);
    nonceBytes[3] = static_cast<uint8_t>((n0 >> 24) & 0xFF);
    nonceBytes[4] = static_cast<uint8_t>(n1 & 0xFF);
    nonceBytes[5] = static_cast<uint8_t>((n1 >> 8) & 0xFF);
    nonceBytes[6] = static_cast<uint8_t>((n1 >> 16) & 0xFF);
    nonceBytes[7] = static_cast<uint8_t>((n1 >> 24) & 0xFF);
    uint64_t ctr = (static_cast<uint64_t>(input[13]) << 32) | static_cast<uint64_t>(input[12]);
    uint8_t zeros[64] = {0};
    // Use Monocypher's ChaCha20 DJB variant to produce the keystream block.
    crypto_chacha20_djb(referenceOutput, zeros, 64, keyBytes, nonceBytes, ctr);

    const auto chachaOutput = rngChaCha.block();
    REQUIRE(std::memcmp(referenceOutput, chachaOutput.data(), 64) == 0);
  }
}

namespace {

using ChaCha20 = simdrng::ChaCha<20>;

constexpr std::array<ChaCha20::matrix_word, 8> kStreamKey = {0x01020304, 0x05060708, 0x09101112, 0x13141516,
                                                             0x17181920, 0x21222324, 0x25262728, 0x29303132};

} // namespace

// stream() must be usable in constant expressions.
static_assert(ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{3})() !=
              ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{4})());

namespace {

// detail::widths_counter128 (node 16 / gpu 16 / thread 32, block 2^64): the
// jump lands in the nonce (state words 14-15), the high half of the 128-bit
// block position. Thread sits at nonce bit 0, gpu at 32, node at 48.
constexpr std::uint64_t kThreadStride = std::uint64_t{1};
constexpr std::uint64_t kGpuStride = std::uint64_t{1} << 32;
constexpr std::uint64_t kNodeStride = std::uint64_t{1} << 48;
constexpr std::uint64_t kStreamNonce = 0x1122334455667788ULL;

} // namespace

TEST_CASE("ChaCha nonce words hold the levels and the key and counter stay put", "[chacha][stream]") {
  for (const std::uint64_t n : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{5}, std::uint64_t{0xFFFF}}) {
    for (const std::uint64_t g : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{5}, std::uint64_t{0xFFFF}}) {
      for (const std::uint64_t t : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{8}, std::uint64_t{0xFFFFFFFF}}) {
        INFO("node: " << n << " gpu: " << g << " thread: " << t);
        const auto st = ChaCha20{kStreamKey, 0, kStreamNonce}
                            .stream(simdrng::node{n}, simdrng::gpu{g}, simdrng::thread{t})
                            .getState();
        // distance = n * 2^48 + g * 2^32 + t in the nonce; the counter stays put.
        const std::uint64_t nonce = kStreamNonce + (n << 48) + (g << 32) + t;
        REQUIRE(st[12] == 0);
        REQUIRE(st[13] == 0);
        REQUIRE(st[14] == static_cast<std::uint32_t>(nonce));
        REQUIRE(st[15] == static_cast<std::uint32_t>(nonce >> 32));
        for (int i = 0; i < 8; ++i) {
          REQUIRE(st[4 + i] == kStreamKey[static_cast<std::size_t>(i)]);
        }
      }
    }
  }
}

TEST_CASE("ChaCha stream(level{v}) equals the plain generator at position v * 2^field blocks", "[chacha][stream]") {
  for (const std::uint64_t v : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{7}}) {
    INFO("v: " << v);
    REQUIRE(stream_level_contract::draw64(ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{v})) ==
            stream_level_contract::draw64(ChaCha20{kStreamKey, 0, v * kThreadStride}));
    REQUIRE(stream_level_contract::draw64(ChaCha20{kStreamKey, 0, 0}.stream(simdrng::gpu{v})) ==
            stream_level_contract::draw64(ChaCha20{kStreamKey, 0, v * kGpuStride}));
    REQUIRE(stream_level_contract::draw64(ChaCha20{kStreamKey, 0, 0}.stream(simdrng::node{v})) ==
            stream_level_contract::draw64(ChaCha20{kStreamKey, 0, v * kNodeStride}));
  }
}

TEST_CASE("ChaCha one-level streams compose into the tuple stream", "[chacha][stream]") {
  const auto composed =
      ChaCha20{kStreamKey, 0, 0}.stream(simdrng::node{5}).stream(simdrng::gpu{6}).stream(simdrng::thread{7});
  const auto tuple = ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{7}, simdrng::node{5}, simdrng::gpu{6});
  REQUIRE(stream_level_contract::draw64(composed) == stream_level_contract::draw64(tuple));
}

TEST_CASE("ChaCha stream draws only advance the counter", "[chacha][stream]") {
  for (const std::uint64_t t : {std::uint64_t{0}, std::uint64_t{3}, std::uint64_t{0x80000003ULL}}) {
    INFO("t: " << t);
    auto s = ChaCha20{kStreamKey, 0, 0}.stream(simdrng::node{5}, simdrng::gpu{6}, simdrng::thread{t});
    for (int k = 0; k < 16; ++k) {
      s();
    }
    // 16 outputs = 2 blocks: counter 2 past the stream start, nonce unchanged.
    const auto st = s.getState();
    const std::uint64_t nonce = (std::uint64_t{5} << 48) + (std::uint64_t{6} << 32) + t;
    REQUIRE(st[12] == 2);
    REQUIRE(st[13] == 0);
    REQUIRE(st[14] == static_cast<std::uint32_t>(nonce));
    REQUIRE(st[15] == static_cast<std::uint32_t>(nonce >> 32));
  }
}

TEST_CASE("ChaCha adjacent thread streams share no value", "[chacha][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    const auto a = stream_level_contract::draw64(ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{i}));
    const auto b = stream_level_contract::draw64(ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{i + 1}));
    int shared = 0;
    for (const auto xv : a) {
      for (const auto yv : b) {
        shared += (xv == yv);
      }
    }
    REQUIRE(shared == 0);
  }
}

TEST_CASE("ChaCha adjacent streams do not overlap at the block boundary", "[chacha][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{0x7FFFFFFF}}) {
    INFO("i: " << i);
    // Stream i owns the 2^64 counter blocks of nonce i. The boundary blocks
    // are its last block (position (i + 1) * 2^64 - 1) and the first block of
    // stream i + 1 (position (i + 1) * 2^64).
    const auto last = ChaCha20{kStreamKey, ~std::uint64_t{0}, i}.block();
    const auto first_next = ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{i + 1}).block();
    const auto first = ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{i}).block();
    REQUIRE(last != first_next);
    REQUIRE(first != first_next);
    REQUIRE(first != last);
    // The boundary blocks equal the plain generator at their positions.
    REQUIRE(first_next == ChaCha20{kStreamKey, 0, i + 1}.block());
    REQUIRE(last == ChaCha20{kStreamKey, ~std::uint64_t{0}, 0}.stream(simdrng::thread{i}).block());
  }
}

TEST_CASE("chacha max in-range levels work and stay distinct", "[chacha][stream]") {
  // node 16 bits (0xFFFF), gpu 16 bits (0xFFFF) and thread 32 bits
  // (0xFFFFFFFF) are the largest values that fit their fields.
  constexpr std::uint64_t max_nonce = (std::uint64_t{0xFFFF} << 48) | (std::uint64_t{0xFFFF} << 32) | 0xFFFFFFFFULL;
  const auto maxed =
      ChaCha20{kStreamKey, 0, 0}.stream(simdrng::node{0xFFFF}, simdrng::gpu{0xFFFF}, simdrng::thread{0xFFFFFFFFULL});
  REQUIRE(stream_level_contract::draw64(maxed) == stream_level_contract::draw64(ChaCha20{kStreamKey, 0, max_nonce}));
  const auto below =
      ChaCha20{kStreamKey, 0, 0}.stream(simdrng::node{0xFFFF}, simdrng::gpu{0xFFFF}, simdrng::thread{0xFFFFFFFEULL});
  REQUIRE(stream_level_contract::draw64(maxed) != stream_level_contract::draw64(below));
}

TEST_CASE("ChaCha stream level contract", "[chacha][stream]") {
  stream_level_contract::check_stream_levels<ChaCha20>(
      [](auto... levels) { return ChaCha20{kStreamKey, 0, 0}.stream(levels...); });
}

TEMPLATE_TEST_CASE("ChaCha zero tags equal omitted tags after a draw", "[chacha][stream]", simdrng::ChaCha<8>,
                   simdrng::ChaCha<12>, simdrng::ChaCha<20>) {
  // 8 results of 64 bits per block: k = 1, 4 (inside one block), 9 (past one boundary).
  stream_level_contract::check_zero_tags_after_draw(TestType{kStreamKey, 0, 0}, {1, 4, 9});
}
