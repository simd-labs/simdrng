#include <simdrng/macros.hpp>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace simdrng {
namespace {

// Bit-by-bit reference: no shift by the bit width, no shared code with rotl.
template <class T> constexpr T ref_rotl(T x, int k) {
  constexpr int w = static_cast<int>(sizeof(T) * 8);
  k %= w;
  if (k < 0) {
    k += w;
  }
  T r = 0;
  for (int i = 0; i < w; ++i) {
    if ((x >> i) & T{1}) {
      r |= static_cast<T>(T{1} << ((i + k) % w));
    }
  }
  return r;
}

// k == 0 exercises the edge where a naive (x << k) | (x >> (width - k)) rotl
// shifts by the full bit width: undefined behavior caught as a hard error in
// constant evaluation.
static_assert(rotl(std::uint32_t{0xDEADBEEF}, 0) == 0xDEADBEEFu);
static_assert(rotl(std::uint64_t{0xDEADBEEFCAFEBABE}, 0) == 0xDEADBEEFCAFEBABEull);
static_assert(rotl(std::uint32_t{0x80000000}, 1) == 0x00000001u);
static_assert(rotl(std::uint32_t{0x00000001}, 31) == 0x80000000u);
static_assert(rotl(std::uint64_t{1}, 1) == 2ull);
static_assert(rotl(std::uint64_t{1}, 63) == 0x8000000000000000ull);

template <class T> void check_against_reference() {
  constexpr int w = static_cast<int>(sizeof(T) * 8);
  const T values[] = {T{0}, T{1}, static_cast<T>(~T{0}), static_cast<T>(T{1} << (w - 1)),
                      static_cast<T>(0xDEADBEEFCAFEBABEull)};
  for (T x : values) {
    for (int k : {0, 1, w - 1, w / 3, w}) {
      REQUIRE(rotl(x, k) == ref_rotl(x, k));
    }
  }
}

} // namespace

TEST_CASE("rotl: uint32_t matches bit-by-bit reference") { check_against_reference<std::uint32_t>(); }

TEST_CASE("rotl: uint64_t matches bit-by-bit reference") { check_against_reference<std::uint64_t>(); }

} // namespace simdrng
