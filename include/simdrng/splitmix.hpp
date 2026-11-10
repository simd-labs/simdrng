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

#include <cstdint>

#include "macros.hpp"

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
   * @brief Per-thread stream constructor. Thread @p stream_index owns the 2^32
   * consecutive values starting at output stream_index * 2^32 + 1 of the seed
   * stream (state = seed + i * 2^32 * gamma). O(1).
   * @param state The initial 64-bit seed state.
   * @param stream_index The stream (thread) index.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE static constexpr SplitMix stream(const std::uint64_t state,
                                                                    const std::uint64_t stream_index) noexcept {
    // Streams are distinct for stream_index < 2^32 (the Weyl orbit is 2^64 long).
    return SplitMix{state + (stream_index << 32) * GAMMA};
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
