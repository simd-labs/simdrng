#pragma once

#include <cstdint>

// MSVC has no __uint128_t, which Vigna's MWC64X/32 C reference uses once, in
// mulmod: (__uint128_t)a * b % mod. This minimal 128-bit type keeps the
// reference compiling unmodified. Included by test_mwc64x32.cpp and
// test_mwc64x32_mulmod.cpp when __int128 is unavailable or SIMDRNG_TEST_U128_SHIM
// forces the fallback path.
namespace mwc_ref {
struct u128 {
  std::uint64_t hi{0};
  std::uint64_t lo{0};
  constexpr u128() = default;
  constexpr explicit u128(const std::uint64_t v) noexcept : hi(0), lo(v) {}
  constexpr u128(const std::uint64_t hi, const std::uint64_t lo) noexcept : hi(hi), lo(lo) {}
  // a * b as a 128-bit product (hi, lo), by 4 half-64-bit multiplications.
  friend constexpr u128 operator*(const u128 &a, const std::uint64_t b) noexcept {
    constexpr std::uint64_t M32 = 0xFFFFFFFFULL;
    const std::uint64_t al = a.lo & M32, ah = a.lo >> 32;
    const std::uint64_t bl = b & M32, bh = b >> 32;
    const std::uint64_t p0 = al * bl;
    const std::uint64_t d0 = p0 & M32;
    std::uint64_t w = al * bh;
    std::uint64_t carry = (w + (p0 >> 32)) < w;
    w += p0 >> 32;
    carry += (w + ah * bl) < w;
    w += ah * bl;
    // w holds the middle sum mod 2^64; the wrap lost 2^64, worth 2^32 of (sum >> 32).
    return {ah * bh + (w >> 32) + (carry << 32), d0 | ((w & M32) << 32)};
  }
  // (hi * 2^64 + lo) % d by binary long division (shift-subtract).
  friend constexpr std::uint64_t operator%(const u128 v, const std::uint64_t d) noexcept {
    std::uint64_t r = 0;
    for (std::uint64_t i = 0; i < 128; ++i) {
      const std::uint64_t bit = (i < 64) ? ((v.hi >> (63 - i)) & 1U) : ((v.lo >> (127 - i)) & 1U);
      const std::uint64_t rhi = r >> 63;
      r = (r << 1) | bit;
      if (rhi != 0U || r >= d) {
        r -= d;
      }
    }
    return r;
  }
};
} // namespace mwc_ref
