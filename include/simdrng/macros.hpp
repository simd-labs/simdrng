#pragma once

#if defined(_MSC_VER)
#define SIMDRNG_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define SIMDRNG_ALWAYS_INLINE inline
#endif

// Lambda call-operators can't take the `inline` keyword, so force-inlining a
// lambda body (e.g. a poet::static_for callable) needs the bare attribute. GCC
// otherwise outlines and clones heavy generic-lambda bodies per instantiation.
#if defined(_MSC_VER)
#define SIMDRNG_ALWAYS_INLINE_LAMBDA [[msvc::forceinline]]
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_ALWAYS_INLINE_LAMBDA __attribute__((always_inline))
#else
#define SIMDRNG_ALWAYS_INLINE_LAMBDA
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_FLATTEN __attribute__((flatten))
#else
#define SIMDRNG_FLATTEN
#endif

#if defined(_MSC_VER)
#define SIMDRNG_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_RESTRICT __restrict__
#else
#define SIMDRNG_RESTRICT
#endif

#if defined(_MSC_VER)
#define SIMDRNG_NEVER_INLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_NEVER_INLINE __attribute__((cold, noinline))
#else
#define SIMDRNG_NEVER_INLINE
#endif

// Host+device callable marker for the scalar generator headers; empty where no
// device compiler is in play. HIP reuses it unchanged.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SIMDRNG_HD __host__ __device__
#else
#define SIMDRNG_HD
#endif

// Device-safe spellings for std:: constexpr pieces. Under nvcc a constexpr
// __host__ function called from device code compiles to a silent trap, so the
// device paths must use the CCCL equivalents (shipped with the CUDA toolkit).
#include <array>
#include <bit>
#include <cstddef>
#include <type_traits>
#if defined(__CUDACC__) && defined(__CUDACC_VER_MAJOR__)
#include <cuda/std/bit>
#endif
// A function template, not a macro: template arguments carry commas.
template <class To, class From> SIMDRNG_HD constexpr To simdrng_bit_cast(const From &v) noexcept {
#if defined(__CUDACC__) && defined(__CUDACC_VER_MAJOR__)
  return ::cuda::std::bit_cast<To>(v);
#else
  return ::std::bit_cast<To>(v);
#endif
}

namespace simdrng {
/**
 * @brief Rotates the bits of an unsigned integer to the left.
 * @tparam T Unsigned integer type.
 * @param x The value to rotate.
 * @param k The number of bits to rotate left.
 * @return The rotated value. Defined for all @p k; the count is taken modulo
 *         the bit width.
 */
template <class T> SIMDRNG_HD constexpr T rotl(T x, int k) noexcept {
  static_assert(std::is_unsigned_v<T>, "rotl requires unsigned integral type");
  constexpr int width = static_cast<int>(sizeof(T) * 8);
  k &= (width - 1);
  return (x << k) | (x >> ((width - k) & (width - 1)));
}
} // namespace simdrng

// One array type for every translation unit, nvcc or host. std::array's
// constexpr members are host-only under strict nvcc (a runtime call traps),
// and a cuda::std::array alias mangles as a different type than std::array,
// which is an ODR hazard in mixed nvcc/host builds. Layout matches
// std::array<T, N>, so simdrng_bit_cast converts at std::array boundaries.
namespace simdrng {
template <class T, std::size_t N> struct array {
  using value_type = T;
  using size_type = std::size_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = T *;
  using const_iterator = const T *;

  T m_data[N];

  SIMDRNG_HD constexpr reference operator[](size_type i) noexcept { return m_data[i]; }
  SIMDRNG_HD constexpr const_reference operator[](size_type i) const noexcept { return m_data[i]; }
  SIMDRNG_HD static constexpr size_type size() noexcept { return N; }
  SIMDRNG_HD constexpr pointer data() noexcept { return m_data; }
  SIMDRNG_HD constexpr const_pointer data() const noexcept { return m_data; }
  SIMDRNG_HD constexpr iterator begin() noexcept { return m_data; }
  SIMDRNG_HD constexpr const_iterator begin() const noexcept { return m_data; }
  SIMDRNG_HD constexpr iterator end() noexcept { return m_data + N; }
  SIMDRNG_HD constexpr const_iterator end() const noexcept { return m_data + N; }
};

template <class T, std::size_t N>
SIMDRNG_HD constexpr bool operator==(const array<T, N> &a, const array<T, N> &b) noexcept {
  for (std::size_t i = 0; i < N; ++i) {
    if (!(a[i] == b[i])) {
      return false;
    }
  }
  return true;
}
} // namespace simdrng

// simdrng ships as a STATIC library (and header-only public API). The per-arch
// dispatch-functor instantiations and the out-of-line XoshiroSIMD ctor need
// external linkage so the explicit instantiations resolve at static-link time,
// but they are implementation details and must NOT be re-exported from a shared
// object that links the static lib (e.g. the Python extension, which should
// export only its module init). Mark them hidden; MSVC static libs don't export
// without dllexport, so nothing is needed there.
#if defined(__GNUC__) || defined(__clang__)
#define SIMDRNG_LOCAL __attribute__((visibility("hidden")))
#else
#define SIMDRNG_LOCAL
#endif

// Runtime SIMD dispatch. When 1, the *SIMD types dispatch over the compiled
// per-arch tiers (portable, set by the CMake SIMDRNG_DISPATCH option, which
// propagates this via a PUBLIC compile definition). Default 0: no tiers are
// built and the default aliases resolve to the native (xsimd::best_arch) types.
#ifndef SIMDRNG_DISPATCH
#define SIMDRNG_DISPATCH 0
#endif

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64)
#define SIMDRNG_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SIMDRNG_ARCH_AARCH64 1
#elif defined(__riscv) && (__riscv_xlen == 64)
#define SIMDRNG_ARCH_RISCV64 1
#endif
