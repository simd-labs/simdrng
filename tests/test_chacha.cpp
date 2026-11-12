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

TEST_CASE("ChaCha stream nonce halves hold the gpu and thread levels", "[chacha][stream]") {
  for (const std::uint64_t g : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{5}, std::uint64_t{1} << 20}) {
    for (const std::uint64_t t : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{8}, std::uint64_t{1} << 20}) {
      INFO("gpu: " << g << " thread: " << t);
      const auto st = ChaCha20{kStreamKey, 0, 0}.stream(simdrng::gpu{g}, simdrng::thread{t}).getState();
      REQUIRE(st[12] == 0);
      REQUIRE(st[13] == 0);
      REQUIRE(st[14] == static_cast<std::uint32_t>(t));
      REQUIRE(st[15] == static_cast<std::uint32_t>(g));
    }
  }
}

TEST_CASE("ChaCha stream draws only advance the counter", "[chacha][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{3}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    auto s = ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{i});
    for (int k = 0; k < 16; ++k) {
      s();
    }
    // 16 outputs = 2 blocks: counter word 2, nonce words unchanged.
    const auto st = s.getState();
    REQUIRE(st[12] == 2);
    REQUIRE(st[13] == 0);
    REQUIRE(st[14] == static_cast<std::uint32_t>(i));
    REQUIRE(st[15] == 0);
  }
}

TEST_CASE("ChaCha adjacent thread streams share no value", "[chacha][stream]") {
  for (const std::uint64_t i : {std::uint64_t{0}, std::uint64_t{1}, std::uint64_t{1} << 20}) {
    INFO("i: " << i);
    auto a = ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{i});
    auto b = ChaCha20{kStreamKey, 0, 0}.stream(simdrng::thread{i + 1});
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

TEST_CASE("ChaCha stream level contract", "[chacha][stream]") {
  stream_level_contract::check_stream_levels<ChaCha20>(
      [](auto... levels) { return ChaCha20{kStreamKey, 0, 0}.stream(levels...); });
}

TEST_CASE("chacha max in-range levels match pinned head output", "[chacha][stream]") {
  // gpu and thread each hold one 32-bit nonce half (0xFFFFFFFF), the largest
  // values that fit. The draws are pinned to the output of HEAD 911f82b at
  // key kStreamKey; the release range check must not move any in-range output.
  constexpr std::uint64_t expected[] = {0x598b8890720aeaddULL, 0xb24705df9756a5e1ULL, 0xa955dfa46e2f6ccbULL,
                                        0xb4b4d208476e53c6ULL};
  auto g = ChaCha20{kStreamKey, 0, 0}.stream(simdrng::gpu{0xFFFFFFFFULL}, simdrng::thread{0xFFFFFFFFULL});
  for (const auto want : expected) {
    REQUIRE(g() == want);
  }
}
