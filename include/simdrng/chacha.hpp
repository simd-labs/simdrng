#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <limits>

#include "macros.hpp"
#include "stream_levels.hpp"

namespace simdrng {

/**
 * @class ChaCha
 * @brief ChaCha stream-cipher core repurposed as a counter-based RNG.
 *
 * Each 64-byte block is produced by running @p R rounds of the ChaCha
 * quarter-round over a 4x4 state matrix built from a 256-bit key, a 64-bit
 * counter and a 64-bit nonce, then adding the original matrix. Because the
 * block is a pure function of (key, counter, nonce), streams are reproducible
 * and seekable: bumping the counter selects the next non-overlapping block.
 *
 * @tparam R Number of rounds (default 20, i.e. ChaCha20; always rounded up to
 *           an even number since odd and even rounds are issued in pairs).
 */
template <std::uint8_t R = 20> class ChaCha {

protected:
  static constexpr auto MATRIX_WORDCOUNT = std::uint8_t{16};
  static constexpr auto KEY_WORDCOUNT = std::uint8_t{8};

public:
  using result_type = std::uint64_t;
  using input_word = std::uint64_t;
  using matrix_word = std::uint32_t;
  using matrix_type = std::array<matrix_word, MATRIX_WORDCOUNT>;
  using result_cache_type = std::array<result_type, MATRIX_WORDCOUNT / 2>;

  /**
   * @brief Smallest value operator() can return.
   * @return 0.
   */
  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE auto(min)() noexcept {
    return (std::numeric_limits<result_type>::min)();
  }

  /**
   * @brief Largest value operator() can return.
   * @return 2^64 - 1.
   */
  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE auto(max)() noexcept {
    return (std::numeric_limits<result_type>::max)();
  }

  /**
   * @brief Construct a scalar ChaCha generator with given key, counter and nonce
   * @param key A 256-bit key, divided up into eight 32-bit words.
   * @param counter Initial value of the counter.
   * @param nonce Initial value of the nonce.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr explicit ChaCha(const std::array<matrix_word, KEY_WORDCOUNT> &key,
                                                             const input_word counter,
                                                             const input_word nonce) noexcept {
    // First four words (i.e. top-row) are always the same constants
    // They spell out "expand 2-byte k" in ASCII (little-endian)
    m_state[0] = 0x61707865;
    m_state[1] = 0x3320646e;
    m_state[2] = 0x79622d32;
    m_state[3] = 0x6b206574;

    const auto key_words = simdrng_bit_cast<simdrng::array<matrix_word, KEY_WORDCOUNT>>(key);
    for (auto i = 0; i < 8; ++i) {
      m_state[4 + i] = key_words[i];
    }

    // ChaCha assumes little-endianness
    m_state[12] = static_cast<matrix_word>(counter & 0xFFFFFFFF);
    m_state[13] = static_cast<matrix_word>(counter >> 32);
    m_state[14] = static_cast<matrix_word>(nonce & 0xFFFFFFFF);
    m_state[15] = static_cast<matrix_word>(nonce >> 32);
  }

  /**
   * @brief Level constructor: ChaCha(key, counter, nonce, levels...) equals
   * ChaCha{key, counter, nonce}.stream(levels...).
   * @param key A 256-bit key, divided up into eight 32-bit words.
   * @param counter Initial value of the counter.
   * @param nonce Initial value of the nonce.
   * @param levels Any subset of {node, gpu, thread}, each at most once.
   */
  template <typename... Levels>
    requires(sizeof...(Levels) > 0 && detail::all_levels_v<Levels...> && detail::unique_levels_v<Levels...>)
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr explicit ChaCha(const std::array<matrix_word, KEY_WORDCOUNT> &key,
                                                             const input_word counter, const input_word nonce,
                                                             Levels... levels) noexcept
      : ChaCha(ChaCha{key, counter, nonce}.stream(levels...)) {}

  /**
   * @brief Hierarchical stream selection. gpu takes the high nonce half and
   * thread the low nonce half (32 bits each, an out-of-range value fails hard
   * in every build type), so streams are disjoint by construction for a given
   * key and every stream keeps the full 64-bit counter range. node{n} XORs a
   * SplitMix expansion of n into the eight key words (node{0} is the
   * identity). O(R).
   * @param levels Any subset of {node, gpu, thread}, in any order, each at most once.
   * @return The generator for the addressed stream.
   */
  template <typename... Levels>
    requires(detail::all_levels_v<Levels...> && detail::unique_levels_v<Levels...>)
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr ChaCha stream(Levels... levels) const noexcept {
    ChaCha g = *this;
    if constexpr (detail::has_level_v<node, Levels...>) {
      const std::uint64_t n = detail::pick_level<node>(levels...).value;
      if (n != 0) {
        std::uint64_t z = n;
        for (auto i = 0; i < KEY_WORDCOUNT; ++i) {
          z = detail::mix64(z);
          g.m_state[4 + i] ^= static_cast<matrix_word>(z);
        }
        // The key changed and the cache position moved into a new stream:
        // drop any cached block so the next draw comes from the new key.
        g.m_result_index = static_cast<std::uint8_t>(g.m_result_cache.size());
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
    assert(gpu_v < (std::uint64_t{1} << 32) && "gpu value overflows its nonce half");
    assert(thread_v < (std::uint64_t{1} << 32) && "thread value overflows its nonce half");
    if (gpu_v >= (std::uint64_t{1} << 32) || thread_v >= (std::uint64_t{1} << 32)) [[unlikely]] {
      detail::stream_level_out_of_range();
    }
    g.m_state[14] = static_cast<matrix_word>(thread_v);
    g.m_state[15] = static_cast<matrix_word>(gpu_v);
    g.m_result_index = static_cast<std::uint8_t>(g.m_result_cache.size());
    return g;
  }

  /**
   * @brief Generates the next 64-bit output.
   * @return The next 64-bit output.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr result_type operator()() noexcept { return next_result(); }

  /**
   * @brief Generates a uniform random number in the range [0, 1).
   * @return A uniform random number.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr double uniform() noexcept {
    return static_cast<double>(operator()() >> 11) * 0x1.0p-53;
  }

  /**
   * @brief Generates the next 64-byte ChaCha block.
   * @return The next 64-byte ChaCha block.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr matrix_type block() noexcept {
    if (m_result_index < m_result_cache.size()) {
      auto cached_block = results_to_block(m_result_cache);
      m_result_index = static_cast<std::uint8_t>(m_result_cache.size());
      return simdrng_bit_cast<matrix_type>(cached_block);
    }
    return simdrng_bit_cast<matrix_type>(next_block());
  }

  /**
   * @brief Returns the state of the generator; a 4x4 matrix.
   * @return State of the generator.
   */
  SIMDRNG_HD SIMDRNG_ALWAYS_INLINE constexpr matrix_type getState() const noexcept {
    auto state = m_state;
    if (m_result_index < m_result_cache.size()) {
      const input_word counter = (static_cast<input_word>(state[13]) << 32) | static_cast<input_word>(state[12]);
      const input_word current_counter = counter - 1;
      state[12] = static_cast<matrix_word>(current_counter & 0xFFFFFFFF);
      state[13] = static_cast<matrix_word>(current_counter >> 32);
    }
    return simdrng_bit_cast<matrix_type>(state);
  }

private:
  // Internal state is the one device-safe array type; matrix_type and
  // result_cache_type stay std::array at the public boundary (as at 85cc841).
  using state_type = simdrng::array<matrix_word, MATRIX_WORDCOUNT>;
  using cache_type = simdrng::array<result_type, MATRIX_WORDCOUNT / 2>;

  state_type m_state;
  cache_type m_result_cache{};
  std::uint8_t m_result_index = static_cast<std::uint8_t>(m_result_cache.size());

  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE auto rotl(const matrix_word x, const int k) noexcept {
    return simdrng::rotl(x, k);
  }

  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE void quarter_round(state_type &m, const unsigned int a,
                                                                       const unsigned int b, const unsigned int c,
                                                                       const unsigned int d) noexcept {
    m[a] += m[b];
    m[d] ^= m[a];
    m[d] = rotl(m[d], 16);
    m[c] += m[d];
    m[b] ^= m[c];
    m[b] = rotl(m[b], 12);
    m[a] += m[b];
    m[d] ^= m[a];
    m[d] = rotl(m[d], 8);
    m[c] += m[d];
    m[b] ^= m[c];
    m[b] = rotl(m[b], 7);
  }

  SIMDRNG_HD constexpr SIMDRNG_ALWAYS_INLINE void inc_counter() noexcept {
    if (++m_state[12] == 0) {
      ++m_state[13];
    }
  }

  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE cache_type block_to_results(const state_type &block) noexcept {
    return simdrng_bit_cast<cache_type>(block);
  }

  SIMDRNG_HD static constexpr SIMDRNG_ALWAYS_INLINE state_type results_to_block(const cache_type &results) noexcept {
    return simdrng_bit_cast<state_type>(results);
  }

  SIMDRNG_HD constexpr SIMDRNG_ALWAYS_INLINE result_type next_result() noexcept {
    if (m_result_index >= m_result_cache.size()) [[unlikely]] {
      m_result_cache = block_to_results(next_block());
      m_result_index = 0;
    }
    return m_result_cache[m_result_index++];
  }

  /**
   * @brief Returns the next output from the generator, then increases state's counter by 1.
   * @return The output for the current internal state.
   */
  SIMDRNG_HD SIMDRNG_FLATTEN constexpr SIMDRNG_ALWAYS_INLINE state_type next_block() noexcept {
    state_type x = m_state;

    // Note that we perform both an odd and even round at the same time.
    // As a result the amount of rounds performed is always rounded up to an even number.
    for (auto i = 0; i < R; i += 2) {
      // Odd round
      quarter_round(x, 0, 4, 8, 12);
      quarter_round(x, 1, 5, 9, 13);
      quarter_round(x, 2, 6, 10, 14);
      quarter_round(x, 3, 7, 11, 15);

      // Even round
      quarter_round(x, 0, 5, 10, 15);
      quarter_round(x, 1, 6, 11, 12);
      quarter_round(x, 2, 7, 8, 13);
      quarter_round(x, 3, 4, 9, 14);
    }

    for (auto i = 0; i < MATRIX_WORDCOUNT; ++i) {
      x[i] += m_state[i];
    }

    inc_counter();

    return x;
  }
};

} // namespace simdrng
