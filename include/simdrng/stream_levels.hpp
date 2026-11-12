#pragma once

// Hierarchical stream levels (API approved 2026-11-10). Three tag types,
// ordered outermost to innermost: simdrng::node (machine) > simdrng::gpu
// (device) > simdrng::thread (thread). Each generator with a stream interface
// exposes a const member template stream(levels...) returning a new generator
// and a level constructor Gen{seed}.stream(levels...) == Gen(seed, levels...).
// Levels accept any subset in any order; each type at most once; a missing
// level is 0.
//
// A level tuple selects a position in the one sequence the seed defines.
// Levels never touch the seed or the key. Each generator maps the tuple to
// one jump distance: distance = node * 2^a + gpu * 2^b + thread * 2^c with
// disjoint bit fields (a > b > c), so the mapping is injective. Each stream
// owns a block of 2^c outputs (counter blocks for the counter-based
// generators) and no two tuples of one seed overlap. The width table in
// detail::stream_widths is the one source of truth for the field widths.
// A level value past its field stops the program in every build type
// (detail::stream_level_out_of_range): a compile error in constant
// evaluation, std::abort on host, __trap() on device. No aliasing in any
// build. The base position of the plain generator must stay below the block
// length for the blocks to stay disjoint.

#include <cassert>
#include <cstdint>
#include <cstdlib>
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

/// Per-generator stream-level field widths in bits: the one source of truth.
/// The fields sit at disjoint shifts: thread at block_bits (shift c), gpu at
/// block_bits + thread_bits (shift b), node at block_bits + thread_bits +
/// gpu_bits (shift a). Every stream owns 2^block_bits outputs (counter blocks
/// for the counter-based generators). One row per period or counter budget.
/// A stream that draws more than its length runs into the next stream's
/// numbers, and nothing checks for this. ChaCha wraps to its own start
/// instead, because its counter does not carry into the nonce:
///
///   widths_tight        node 8 / gpu 8 / thread 24, block 2^24.
///     SplitMix64 (period 2^64): 2^24 outputs per stream. A stream that
///     draws more than 2^24 outputs runs into the next stream's numbers.
///     MWC64X32 (period about 2^63): 2^24 outputs per stream. A stream that
///     draws more than 2^24 outputs runs into the next stream's numbers.
///     Philox2x32 (64-bit counter): 2^24 counter blocks per stream. A stream
///     that draws more than 2^24 counter blocks runs into the next stream's
///     numbers.
///   widths_xoshiro256   node 64 / gpu 32 / thread 32, block 2^127.
///     XoshiroScalar (period 2^256 - 1): 2^127 outputs per stream. The
///     fields stop at 255 bits so no block wraps the period. A stream that
///     draws more than 2^127 outputs runs into the next stream's numbers.
///   widths_counter128   node 16 / gpu 16 / thread 32, block 2^64.
///     Philox4x32, Philox2x64 (128-bit counters): 2^64 counter blocks per
///     stream, the full 64-bit counter range as position. A stream that
///     draws more than 2^64 counter blocks runs into the next stream's
///     numbers. ChaCha (64-bit counter plus 64-bit nonce): 2^64 counter
///     blocks per stream. A stream that draws more than 2^64 counter blocks
///     repeats its own numbers.
///   widths_counter256   node 64 / gpu 64 / thread 64, block 2^64.
///     Philox4x64 (256-bit counter): 2^64 counter blocks per stream. A
///     stream that draws more than 2^64 counter blocks runs into the next
///     stream's numbers.
struct stream_widths {
  std::uint64_t node_bits;
  std::uint64_t gpu_bits;
  std::uint64_t thread_bits;
  std::uint64_t block_bits;
  SIMDRNG_HD constexpr std::uint64_t thread_shift() const noexcept { return block_bits; }
  SIMDRNG_HD constexpr std::uint64_t gpu_shift() const noexcept { return block_bits + thread_bits; }
  SIMDRNG_HD constexpr std::uint64_t node_shift() const noexcept { return block_bits + thread_bits + gpu_bits; }
};

inline constexpr stream_widths widths_tight{8, 8, 24, 24};
inline constexpr stream_widths widths_xoshiro256{64, 32, 32, 127};
inline constexpr stream_widths widths_counter128{16, 16, 32, 64};
inline constexpr stream_widths widths_counter256{64, 64, 64, 64};

/// Hard failure for a level value past its field, in every build type. A
/// constant evaluation that reaches the call is a compile error (a
/// non-constexpr call); the host aborts; a CUDA device traps. The stream()
/// functions are noexcept, so no exception path exists.
SIMDRNG_NEVER_INLINE SIMDRNG_HD inline void stream_level_out_of_range() noexcept {
#if defined(__CUDA_ARCH__)
  __trap();
#else
  std::abort();
#endif
}

/// Returns v, or stops the program when v does not fit the field width bits.
/// A 64-bit field holds every std::uint64_t and needs no check.
SIMDRNG_HD constexpr std::uint64_t stream_level_checked(std::uint64_t bits, std::uint64_t v) noexcept {
  assert((bits >= 64 || v < (std::uint64_t{1} << bits)) && "stream level value overflows its field");
  if (bits < 64 && v >= (std::uint64_t{1} << bits)) [[unlikely]] {
    stream_level_out_of_range();
  }
  return v;
}

/// The jump distance for a level tuple in a 64-bit jump space:
/// node * 2^a + gpu * 2^b + thread * 2^c in the field layout of w. One unit is
/// one output (one counter block for the counter-based generators). The
/// fields are disjoint, so the OR is the sum. O(1).
template <typename... Levels>
SIMDRNG_HD constexpr std::uint64_t level_distance64(stream_widths w, Levels... levels) noexcept {
  std::uint64_t distance = 0;
  if constexpr (has_level_v<node, Levels...>) {
    distance |= stream_level_checked(w.node_bits, pick_level<node>(levels...).value) << w.node_shift();
  }
  if constexpr (has_level_v<gpu, Levels...>) {
    distance |= stream_level_checked(w.gpu_bits, pick_level<gpu>(levels...).value) << w.gpu_shift();
  }
  if constexpr (has_level_v<thread, Levels...>) {
    distance |= stream_level_checked(w.thread_bits, pick_level<thread>(levels...).value) << w.thread_shift();
  }
  return distance;
}

} // namespace detail

} // namespace simdrng
