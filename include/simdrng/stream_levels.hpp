#pragma once

// Hierarchical stream levels (API approved 2026-11-10). Three tag types,
// ordered outermost to innermost: simdrng::node (machine) > simdrng::gpu
// (device) > simdrng::thread (thread). Each generator with a stream interface
// exposes a const member template stream(levels...) returning a new generator
// and a level constructor Gen{seed}.stream(levels...) == Gen(seed, levels...).
// Levels accept any subset in any order; each type at most once; a missing
// level is 0. Placement per generator:
//   XoshiroScalar: node folds a SplitMix stream into the state (hash reseed),
//                  gpu = long_jump, thread = jump.
//   Philox4x:      node XORs into the key words, gpu/thread take the high/low
//                  32 bits of the 64-bit stream index in the top counter words.
//   Philox2x32:    node XORs into the key word, gpu gets 8 and thread 24 bits
//                  of the 32-bit index word.
//   ChaCha:        node XORs into the key words, gpu/thread take the two nonce
//                  halves (32 bits each).
//   SplitMix64,    node is a compile error (no jump structure); index bits:
//   MWC64X32:      gpu 8 (MWC 7) / thread 24, debug assert on overflow.

#include <cstdint>
#include <type_traits>

#include "macros.hpp"

namespace simdrng {

/// Outermost stream level: one machine (node) of a run.
struct node {
  std::uint64_t value;
};
/// Middle stream level: one GPU on a node.
struct gpu {
  std::uint64_t value;
};
/// Innermost stream level: one thread on a GPU (or core on a CPU).
struct thread {
  std::uint64_t value;
};

namespace detail {

template <typename T> struct is_level_impl : std::false_type {};
template <> struct is_level_impl<node> : std::true_type {};
template <> struct is_level_impl<gpu> : std::true_type {};
template <> struct is_level_impl<thread> : std::true_type {};

/// True when every type in the pack is one of node, gpu, thread.
template <typename... Levels> inline constexpr bool all_levels_v = (is_level_impl<Levels>::value && ...);

/// Number of occurrences of level type T in the pack.
template <typename T, typename... Levels>
inline constexpr std::size_t level_count_v = (std::size_t{0} + ... + (std::is_same_v<Levels, T> ? 1 : 0));

/// True when no level type appears more than once.
template <typename... Levels>
inline constexpr bool unique_levels_v =
    level_count_v<node, Levels...> <= 1 && level_count_v<gpu, Levels...> <= 1 && level_count_v<thread, Levels...> <= 1;

/// True when level type T appears in the pack.
template <typename T, typename... Levels> inline constexpr bool has_level_v = (std::is_same_v<Levels, T> || ...);

/// Returns the level of type T from the pack, or T{0} when absent.
template <typename T, typename... Levels> SIMDRNG_HD constexpr T pick_level(Levels... levels) noexcept {
  T result{0};
  auto f = [&](auto l) {
    if constexpr (std::is_same_v<decltype(l), T>) {
      result = l;
    }
  };
  (f(levels), ...);
  return result;
}

/// SplitMix64 avalanche over one Weyl step. Used to fold a level value into a
/// key or state word without a reversible bias.
SIMDRNG_HD constexpr std::uint64_t mix64(std::uint64_t z) noexcept {
  z += 0x9e3779b97f4a7c15;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

} // namespace detail

} // namespace simdrng
