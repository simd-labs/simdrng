/*  Written in 2026 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

Ported to C++ by Marco Barbone. Original implementation by Sebastiano Vigna.
*/

#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>

#include "macros.hpp"
#include "splitmix.hpp"
#include "stream_levels.hpp"

namespace simdrng {

/**
 * @class MWC64X32
 * @brief MWC64X/32 — a Marsaglia multiply-with-carry generator with a 32-bit
 * output and a period of approximately 2^63.
 *
 * The state (x, c) simulates a multiplicative LCG modulo 0xfff145f7ffffffff,
 * and the output is x xored with the carry. Each step costs one 64-bit
 * multiplication and one addition, so the generator suits register-scarce
 * environments (e.g. GPUs).
 * Reference implementation: Sebastiano Vigna's MWC64X_32.c (public domain),
 * https://prng.di.unimi.it/MWC64X_32.c
 */
class MWC64X32 {
public:
  using result_type = std::uint32_t;
  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept { return result_type{0}; }
  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept {
    return static_cast<result_type>(~result_type{0});
  }

  /**
   * @brief Constructs the MWC64X32 generator from a 64-bit seed.
   * @param seed The seed value.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr explicit MWC64X32(const std::uint64_t seed) noexcept : m_x{}, m_c(1) {
    // Vigna's seeding rule: c = 1, x = the low 32 bits of the first SplitMix64 output.
    m_x = static_cast<result_type>(SplitMix{seed}());
  }

  /**
   * @brief Level constructor: MWC64X32(seed, levels...) equals
   * MWC64X32{seed}.stream(levels...).
   */
  template <typename... Levels>
    requires(sizeof...(Levels) > 0 && detail::all_levels_v<Levels...> && detail::unique_levels_v<Levels...> &&
             !detail::has_level_v<node, Levels...>)
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr explicit MWC64X32(const std::uint64_t seed, Levels... levels) noexcept
      : MWC64X32(MWC64X32{seed}.stream(levels...)) {}

  /**
   * @brief Hierarchical stream selection. gpu takes 7 and thread 24 bits of
   * the stream index (gpu in the high half, debug assert on overflow), so the
   * index stays below 2^31 and each of the 2^31 streams spans 2^32 values
   * within the ~2^63 period. The generator at index i starts at the base
   * state jumped by i * 2^32 outputs, computed in O(log i) as
   * (A1^(2^32))^i mod MOD. node is rejected: hash the node id into the seed
   * instead. O(1).
   * @param levels Any subset of {gpu, thread}, in any order, each at most once.
   * @return The generator for the addressed stream.
   */
  template <typename... Levels>
    requires(detail::all_levels_v<Levels...> && detail::unique_levels_v<Levels...> &&
             !detail::has_level_v<node, Levels...>)
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr MWC64X32 stream(Levels... levels) const noexcept {
    std::uint64_t gpu_v = 0;
    std::uint64_t thread_v = 0;
    if constexpr (detail::has_level_v<gpu, Levels...>) {
      gpu_v = detail::pick_level<gpu>(levels...).value;
    }
    if constexpr (detail::has_level_v<thread, Levels...>) {
      thread_v = detail::pick_level<thread>(levels...).value;
    }
    assert(gpu_v < (std::uint64_t{1} << 7) && "gpu value overflows its stream-index field");
    assert(thread_v < (std::uint64_t{1} << 24) && "thread value overflows its stream-index field");
    const std::uint64_t index = (gpu_v << 24) | thread_v;
    MWC64X32 g = *this;
    if (index != 0) {
      g.jump_mul(powmod(JUMP_MUL, index));
    }
    return g;
  }

  /**
   * @brief Advances the state and returns the next 32-bit value.
   * @return The next 32-bit output.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept { return next(); }

  /**
   * @brief Returns the state of the generator.
   * @return The state as {x, c}.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr simdrng::array<result_type, 2> getState() const noexcept {
    return {m_x, m_c};
  }

  /**
   * @brief Restores the state, e.g. when resuming a saved stream.
   * @param state The state to restore, as {x, c}.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr void setState(const simdrng::array<result_type, 2> &state) noexcept {
    m_x = state[0];
    m_c = state[1];
  }

  /**
   * @brief Returns the size of the state array.
   * @return The size of the state array.
   */
  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE result_type stateSize() noexcept { return 2; }

  /**
   * @brief Jump function for the generator. It is equivalent to 2^32 calls to next().
   * It can be used to generate 2^31 non-overlapping subsequences for parallel computations.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr void jump() noexcept { jump_mul(JUMP_MUL); }

  /**
   * @brief Long-jump function for the generator. It is equivalent to 2^48 calls to next().
   * It can be used to generate 2^15 starting points, from each of which jump() will
   * generate 2^16 non-overlapping subsequences for parallel distributed computations.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr void long_jump() noexcept { jump_mul(0x7b046c7221442678); }

  /**
   * @brief Arbitrary jump-ahead: advances the state by exactly @p n calls to next().
   * Computes A1^n mod MOD on the fly; for a meaningful jump, n should stay below the
   * period (approximately 2^63). Overload of jump(); the no-argument jump() remains
   * the fixed 2^32 stride.
   * @param n The number of steps to jump ahead.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr void jump(std::uint64_t n) noexcept { jump_mul(powmod(A1, n)); }

private:
  std::uint32_t m_x{};
  std::uint32_t m_c{};

  /// MWC_A1 from Vigna's reference: the inverse of 2^32 modulo MOD.
  static constexpr std::uint64_t A1 = 0xfff145f8ULL;
  /// The prime modulus MOD = A1 * 2^32 - 1 of the simulated LCG.
  static constexpr std::uint64_t MOD = (A1 << 32) - 1ULL;
  /// 2^64 - MOD. addmod folds a wrapped 64-bit sum by this amount (safe because MOD > 2^63).
  static constexpr std::uint64_t TWO64_MINUS_MOD = std::numeric_limits<std::uint64_t>::max() - MOD + 1ULL;
  /// A1^(2^32) mod MOD: the jump() stride multiplier (2^32 steps).
  static constexpr std::uint64_t JUMP_MUL = 0x1012f7168d155184;

  /**
   * @brief Advances the state once. Vigna's step, verbatim.
   * @return The next 32-bit output.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr result_type next() noexcept {
    const result_type result = m_x ^ m_c;
    const std::uint64_t t = A1 * static_cast<std::uint64_t>(m_x) + m_c;
    m_x = static_cast<result_type>(t);
    m_c = static_cast<result_type>(t >> 32);
    return result;
  }

  /**
   * @brief Multiplies the state, read as c * 2^32 + x, by r modulo MOD.
   * Each call to next() multiplies it by A1 modulo MOD (A1 * 2^32 = MOD + 1), so a
   * jump of n steps multiplies it by A1^n mod MOD (Vigna).
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr void jump_mul(const std::uint64_t r) noexcept {
    const std::uint64_t s = mulmod(static_cast<std::uint64_t>(m_c) << 32 | m_x, r);
    m_x = static_cast<result_type>(s);
    m_c = static_cast<result_type>(s >> 32);
  }

  /// a^n mod MOD by square-and-multiply.
  SIMDRNG_HD static SIMDRNG_ALWAYS_INLINE constexpr std::uint64_t powmod(std::uint64_t a, std::uint64_t n) noexcept {
    std::uint64_t r = 1;
    while (n != 0U) {
      if ((n & 1U) != 0U) {
        r = mulmod(r, a);
      }
      n >>= 1;
      if (n != 0U) {
        a = mulmod(a, a);
      }
    }
    return r;
  }

  /// (x + y) mod MOD for x, y <= MOD, without __int128 (MSVC and CUDA device code).
  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE std::uint64_t addmod(const std::uint64_t x,
                                                                         const std::uint64_t y) noexcept {
    std::uint64_t r = x + y;
    if (r < x) {
      r += TWO64_MINUS_MOD; // wrapped: the true sum is r + 2^64, which stays < MOD
    }
    if (r >= MOD) {
      r -= MOD;
    }
    return r;
  }

public:
  /// (a * b) mod MOD for any 64-bit a, b by shift-and-add (Russian peasant multiplication), without __int128.
  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE std::uint64_t mulmod(std::uint64_t a, std::uint64_t b) noexcept {
    // addmod needs operands <= MOD; MOD > 2^63, so one subtraction reduces any 64-bit operand.
    if (a >= MOD) {
      a -= MOD;
    }
    if (b >= MOD) {
      b -= MOD;
    }
    std::uint64_t r = 0;
    std::uint64_t aa = a;
    while (b != 0U) {
      if ((b & 1U) != 0U) {
        r = addmod(r, aa);
      }
      b >>= 1;
      if (b != 0U) {
        aa = addmod(aa, aa);
      }
    }
    return r;
  }
};

} // namespace simdrng
