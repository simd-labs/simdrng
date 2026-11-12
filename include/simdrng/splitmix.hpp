/*  Written in 2014 originally by Guy Steele

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

Ported to C++ by Marco Barbone. Original implementation by Guy Steele.
*/

#pragma once

#include <cassert>
#include <cstdint>

#include "macros.hpp"
#include "stream_levels.hpp"

namespace simdrng {

/**
 * @class SplitMix
 * @brief SplitMix64 — a fast 64-bit splittable generator used to seed the other engines.
 *
 * Given a single 64-bit seed it advances a Weyl sequence (the state is bumped by
 * the golden-ratio constant 0x9e3779b97f4a7c15 each call) and applies an
 * avalanche mix, producing a well-distributed stream of 64-bit values. simdrng
 * uses it to expand a user seed into the larger state of the other generators,
 * so that even low-entropy seeds (e.g. 0 or 1) yield sound starting states.
 */
class SplitMix {
public:
  /**
   * @brief Constructs the generator from a 64-bit seed state.
   * @param state The initial 64-bit state.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr explicit SplitMix(const std::uint64_t state) noexcept : m_state(state) {}

  /**
   * @brief Level constructor: SplitMix(state, levels...) equals
   * SplitMix{state}.stream(levels...).
   */
  template <typename... Levels>
    requires(sizeof...(Levels) > 0 && detail::all_levels_v<Levels...> && detail::unique_levels_v<Levels...>)
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr explicit SplitMix(const std::uint64_t state, Levels... levels) noexcept
      : SplitMix(SplitMix{state}.stream(levels...)) {}

  /**
   * @brief Hierarchical stream selection. node{n} XORs a SplitMix expansion of
   * n into the seed state (node{0} is the identity), the same fold Philox and
   * ChaCha apply to their key words. gpu takes 8 and thread 24 bits of the
   * stream index (gpu in the high half, an out-of-range value fails hard in
   * every build type); the generator at index i owns the 2^32 consecutive
   * values starting at output i * 2^32 + 1 of the seed stream (state + i *
   * 2^32 * gamma), so streams are distinct for index < 2^32. O(1).
   * @param levels Any subset of {node, gpu, thread}, in any order, each at most once.
   * @return The generator for the addressed stream.
   */
  template <typename... Levels>
    requires(detail::all_levels_v<Levels...> && detail::unique_levels_v<Levels...>)
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr SplitMix stream(Levels... levels) const noexcept {
    SplitMix g = *this;
    if constexpr (detail::has_level_v<node, Levels...>) {
      const std::uint64_t n = detail::pick_level<node>(levels...).value;
      if (n != 0) {
        g.m_state ^= detail::mix64(n);
      }
    }
    std::uint64_t gpu_v = 0;
    std::uint64_t thread_v = 0;
    if constexpr (detail::has_level_v<gpu, Levels...>) {
      gpu_v = detail::pick_level<gpu>(levels...).value;
    }
    if constexpr (detail::has_level_v<thread, Levels...>) {
      thread_v = detail::pick_level<thread>(levels...).value;
    }
    assert(gpu_v < (std::uint64_t{1} << 8) && "gpu value overflows its stream-index field");
    assert(thread_v < (std::uint64_t{1} << 24) && "thread value overflows its stream-index field");
    if (gpu_v >= (std::uint64_t{1} << 8) || thread_v >= (std::uint64_t{1} << 24)) [[unlikely]] {
      detail::stream_level_out_of_range();
    }
    const std::uint64_t index = (gpu_v << 24) | thread_v;
    return SplitMix{g.m_state + (index << 32) * GAMMA};
  }

  /**
   * @brief Advances the state and returns the next 64-bit value.
   * @return The next 64-bit output.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr std::uint64_t operator()() noexcept {
    std::uint64_t z = (m_state += GAMMA);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  }

  /**
   * @brief Smallest value operator() can return.
   * @return 0.
   */
  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE std::uint64_t(min)() noexcept { return std::uint64_t{0}; }

  /**
   * @brief Largest value operator() can return.
   * @return 2^64 - 1.
   */
  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE std::uint64_t(max)() noexcept {
    return static_cast<std::uint64_t>(~std::uint64_t{0});
  }

  /**
   * @brief Returns the current 64-bit state.
   * @return The current state.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr std::uint64_t getState() const noexcept { return m_state; }

  /**
   * @brief Restores the state, e.g. when resuming a saved stream.
   * @param state The state to restore.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr void setState(std::uint64_t state) noexcept { m_state = state; }

private:
  /// The golden-ratio increment of the Weyl sequence.
  static constexpr std::uint64_t GAMMA = 0x9e3779b97f4a7c15;

  std::uint64_t m_state;
};

} // namespace simdrng
