// Per-thread independent streams using the stream-level constructor.
#include <cstdio>
#include <simdrng/xoshiro_scalar.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

int main() {
#ifdef _OPENMP
#pragma omp parallel
  {
    // thread{i} jumps the base stream i times: each OpenMP thread owns a
    // disjoint 2^128-value subsequence.
    simdrng::XoshiroScalar rng(42u, simdrng::thread{static_cast<std::uint64_t>(omp_get_thread_num())});
    const auto x = rng();
    std::printf("thread %d: %llu\n", omp_get_thread_num(), static_cast<unsigned long long>(x));
  }
#else
  simdrng::XoshiroScalar rng(42u, simdrng::thread{0});
  std::printf("single thread: %llu\n", static_cast<unsigned long long>(rng()));
#endif
  return 0;
}
