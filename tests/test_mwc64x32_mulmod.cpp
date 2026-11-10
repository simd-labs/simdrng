#include <catch2/catch_all.hpp>
#include <cstdint>
#include <random>
#include <simdrng/mwc64x32.hpp>

#if !defined(__SIZEOF_INT128__) || defined(SIMDRNG_TEST_U128_SHIM)
#include "mwc64x32_u128_shim.hpp"
#endif

namespace {

/// The prime modulus MOD = 0xfff145f8 * 2^32 - 1 of the simulated LCG.
constexpr std::uint64_t MOD = (0xfff145f8ULL << 32) - 1ULL;

/// Edge operands around the reduction boundary at the mulmod entry.
constexpr std::uint64_t kEdge[] = {0ULL, 1ULL, 2ULL, MOD - 1, MOD, MOD + 1, 1ULL << 63, UINT64_MAX - 1, UINT64_MAX};

/// Exact (a * b) mod MOD as a 128-bit product. Uses native __int128 where the
/// compiler has it, otherwise the portable mwc_ref::u128 shim, so the check
/// runs real assertions on every platform instead of skipping.
#if defined(__SIZEOF_INT128__) && !defined(SIMDRNG_TEST_U128_SHIM)
std::uint64_t ref_mulmod(const std::uint64_t a, const std::uint64_t b) {
  __extension__ typedef unsigned __int128 u128;
  return static_cast<std::uint64_t>(static_cast<u128>(a) * b % MOD);
}
#else
std::uint64_t ref_mulmod(const std::uint64_t a, const std::uint64_t b) { return mwc_ref::u128(a) * b % MOD; }
#endif

} // namespace

TEST_CASE("mwc64x32 mulmod matches the 128-bit reference on the edge set", "[mwc64x32][mulmod]") {
  for (const std::uint64_t a : kEdge) {
    for (const std::uint64_t b : kEdge) {
      INFO("a=" << a << " b=" << b);
      REQUIRE(simdrng::MWC64X32::mulmod(a, b) == ref_mulmod(a, b));
    }
  }
}

TEST_CASE("mwc64x32 mulmod matches the 128-bit reference on random pairs", "[mwc64x32][mulmod]") {
  std::mt19937_64 rng{0x243F6A8885A308D3ULL}; // fixed seed for a reproducible run
  for (int i = 0; i < 10000; ++i) {
    const std::uint64_t a = rng();
    const std::uint64_t b = rng();
    INFO("a=" << a << " b=" << b);
    REQUIRE(simdrng::MWC64X32::mulmod(a, b) == ref_mulmod(a, b));
  }
}
