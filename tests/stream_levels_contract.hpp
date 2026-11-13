// Shared checks for the node/gpu/thread stream-level contract: every
// permutation of the passed tags gives a bit-identical stream, an omitted
// level equals that level at 0, distinct values at one level give distinct
// streams, and a repeated tag or a non-level argument is a compile error.
// Each generator test calls check_stream_levels with a make(levels...) stream
// factory. node checks compile only for generators that accept node.
#pragma once

#include <array>
#include <cstdint>

#include <catch2/catch_all.hpp>

#include <simdrng/stream_levels.hpp>

namespace stream_level_contract {

template <class Gen> std::array<std::uint64_t, 64> draw64(Gen g) {
  std::array<std::uint64_t, 64> out{};
  for (auto &v : out) {
    v = g();
  }
  return out;
}

/// True when Gen::stream accepts the node level.
template <class Gen> inline constexpr bool accepts_node_v = requires(const Gen &g) { g.stream(simdrng::node{1}); };

/// True when Gen::stream rejects a repeated tag of type Tag.
template <class Gen, class Tag>
inline constexpr bool rejects_repeated_v = !requires(const Gen &g) { g.stream(Tag{1}, Tag{2}); };

/// True when Gen::stream rejects a non-level argument.
template <class Gen>
inline constexpr bool rejects_non_level_v = !requires(const Gen &g) { g.stream(std::uint64_t{1}); };

template <class Gen, class Make> void check_stream_levels(Make make) {
  static_assert(rejects_repeated_v<Gen, simdrng::thread>, "a repeated tag must not compile");
  static_assert(rejects_non_level_v<Gen>, "a non-level argument must not compile");
  // gpu and thread, every generator: permutations, omitted = 0, distinct.
  REQUIRE(draw64(make(simdrng::gpu{2}, simdrng::thread{3})) == draw64(make(simdrng::thread{3}, simdrng::gpu{2})));
  REQUIRE(draw64(make(simdrng::gpu{2})) == draw64(make(simdrng::gpu{2}, simdrng::thread{0})));
  REQUIRE(draw64(make(simdrng::thread{3})) == draw64(make(simdrng::gpu{0}, simdrng::thread{3})));
  REQUIRE(draw64(make()) == draw64(make(simdrng::gpu{0}, simdrng::thread{0})));
  REQUIRE(draw64(make(simdrng::gpu{1})) != draw64(make(simdrng::gpu{2})));
  REQUIRE(draw64(make(simdrng::thread{1})) != draw64(make(simdrng::thread{2})));
  REQUIRE(draw64(make(simdrng::gpu{1})) != draw64(make(simdrng::thread{1})));
  if constexpr (accepts_node_v<Gen>) {
    static_assert(rejects_repeated_v<Gen, simdrng::node>, "a repeated tag must not compile");
    // node joins the permutation set: all 6 orders of 3 tags, both orders of
    // each node pair.
    const auto ref = draw64(make(simdrng::node{1}, simdrng::gpu{2}, simdrng::thread{3}));
    REQUIRE(draw64(make(simdrng::gpu{2}, simdrng::node{1}, simdrng::thread{3})) == ref);
    REQUIRE(draw64(make(simdrng::node{1}, simdrng::thread{3}, simdrng::gpu{2})) == ref);
    REQUIRE(draw64(make(simdrng::thread{3}, simdrng::node{1}, simdrng::gpu{2})) == ref);
    REQUIRE(draw64(make(simdrng::thread{3}, simdrng::gpu{2}, simdrng::node{1})) == ref);
    REQUIRE(draw64(make(simdrng::gpu{2}, simdrng::thread{3}, simdrng::node{1})) == ref);
    REQUIRE(draw64(make(simdrng::node{1}, simdrng::gpu{2})) == draw64(make(simdrng::gpu{2}, simdrng::node{1})));
    REQUIRE(draw64(make(simdrng::node{1}, simdrng::thread{3})) == draw64(make(simdrng::thread{3}, simdrng::node{1})));
    REQUIRE(draw64(make(simdrng::node{1})) == draw64(make(simdrng::node{1}, simdrng::gpu{0}, simdrng::thread{0})));
    REQUIRE(draw64(make(simdrng::node{1}, simdrng::thread{3})) ==
            draw64(make(simdrng::node{1}, simdrng::gpu{0}, simdrng::thread{3})));
    REQUIRE(draw64(make()) == draw64(make(simdrng::node{0}, simdrng::gpu{0}, simdrng::thread{0})));
    REQUIRE(draw64(make(simdrng::node{1})) != draw64(make(simdrng::node{2})));
    REQUIRE(draw64(make(simdrng::node{1})) != draw64(make(simdrng::thread{1})));
    REQUIRE(draw64(make(simdrng::node{1})) != draw64(make(simdrng::gpu{1})));
  }
}

// After k output calls, stream() with no tags must produce the same next
// outputs as stream() with every tag explicitly zero, on the same consumed
// generator. counts cover a first draw, a position inside one block, and a
// position past one block boundary for the generator under test.
template <class Gen> void check_zero_tags_after_draw(Gen g, std::initializer_list<std::size_t> counts) {
  for (const std::size_t k : counts) {
    INFO("k: " << k);
    Gen consumed = g;
    for (std::size_t i = 0; i < k; ++i) {
      (void)consumed();
    }
    REQUIRE(draw64(consumed.stream()) ==
            draw64(consumed.stream(simdrng::node{0}, simdrng::gpu{0}, simdrng::thread{0})));
  }
}

} // namespace stream_level_contract
