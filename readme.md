# simdrng

[![CI](https://github.com/DiamonDinoia/simdrng/actions/workflows/ci.yml/badge.svg)](https://github.com/DiamonDinoia/simdrng/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/simdrng/badge/?version=latest)](https://simdrng.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/DiamonDinoia/simdrng/branch/main/graph/badge.svg)](https://codecov.io/gh/DiamonDinoia/simdrng)
[![License: BSD-3-Clause-Attribution](https://img.shields.io/badge/License-BSD--3--Clause--Attribution-blue.svg)](LICENSE)

> **Note:** `simdrng` is pre-1.0 — the API may change before the first tagged release.

C++20 random number generators that run scalar **and** SIMD-accelerated, with
nanobind-powered Python bindings that drop straight into NumPy and SciPy.

- **Header-only** (plus one tiny dispatch TU for runtime SIMD selection); the
  scalar path is dependency-free.
- **Four generator families** — Xoshiro256++, SplitMix64, ChaCha, Philox — each
  available scalar, runtime-SIMD-dispatched, and `-march=native`.
- **Drop-in** with `std::uniform_int_distribution` and friends, plus a fast
  `uniform()` returning a `double` in `[0, 1)`.
- **Parallel-ready**: disjoint streams at the `node`, `gpu` and `thread`
  levels, or Philox's counter-based reproducibility.
- **Python** — every generator is a `numpy.random.Generator` with a GIL-released
  C++ bulk-fill fast path.

## Why simdrng

General-purpose PRNGs are written one number at a time, leaving most of a modern
CPU's vector units idle. simdrng pairs best-in-class scalar generators
(Blackman & Vigna's xoshiro256++, the Random123 Philox counter-based design,
the ChaCha cipher core) with hand-tuned SIMD backends, and picks the right
implementation **at runtime** via [xsimd](https://github.com/xtensor-stack/xsimd)
dispatch — so the same binary uses AVX-512 on a server and NEON on an ARM
laptop with no recompilation. When you can commit to a target with
`-march=native`, the `*Native` generators inline the widest available width
directly. The result is NumPy-beating bulk throughput from C++ or Python while
keeping single-value calls as fast as the reference scalar code.

## Quick start

```cpp
#include <simdrng/xoshiro.hpp>

simdrng::Xoshiro rng(42);          // seed
std::uint64_t x = rng();           // next 64-bit value
double u = rng.uniform();          // double in [0, 1)
```

Bulk fill is where SIMD pays off. `XoshiroSIMD` refills an internal cache with
wide SIMD batches, so a plain generate loop runs vectorised under the hood — no
special bulk API to learn:

```cpp
#include <simdrng/xoshiro_simd.hpp>
#include <vector>

simdrng::XoshiroSIMD rng(42);
std::vector<double> out(1'000'000);
for (auto &x : out)
  x = rng.uniform();              // cache refilled with SIMD batches
```

From Python it is a `numpy.random.Generator`:

```python
import simdrng

rng = simdrng.XoshiroSIMD(seed=42)          # a numpy.random.Generator
samples = rng.random(10_000_000)            # GIL-released C++ bulk fill
```

More runnable programs live in [`examples/`](examples/) (C++ and Python), and
ready-to-run [Compiler Explorer links](https://github.com/DiamonDinoia/simdrng/tree/godbolt-links)
are published to the `godbolt-links` branch.

## Generators

| Family               | Scalar | SIMD dispatch | `-march=native` |
|----------------------|:------:|:-------------:|:---------------:|
| Xoshiro256++         | ✅     | ✅            | ✅              |
| SplitMix64           | ✅     | —             | —               |
| ChaCha 8/12/20       | ✅     | ✅            | ✅              |
| Philox 2x/4x × 32/64 | ✅     | ✅            | ✅              |

All generators satisfy the standard `UniformRandomBitGenerator` requirements, so
they compose with `std::uniform_int_distribution` and friends. The `uniform()`
method returns a `double` in `[0, 1)` faster than
`std::uniform_real_distribution`; see <https://prng.di.unimi.it/#remarks> for
the rationale. Per-family guidance is in the
[documentation](https://simdrng.readthedocs.io/en/latest/) (Guides section).

## Parallel streams

The level tags `simdrng::node{...}`, `simdrng::gpu{...}` and
`simdrng::thread{...}` carve out disjoint streams. Each tag holds one
`std::uint64_t value`. Each generator accepts all three tags. All permutations
give the same stream. An omitted level is `0`. A duplicate tag,
or a non-level argument, is a compile error:

```cpp
#include <simdrng/xoshiro_scalar.hpp>

const std::uint64_t seed = 42;
simdrng::XoshiroScalar full{seed, simdrng::node{1}, simdrng::gpu{2}, simdrng::thread{3}};
simdrng::XoshiroScalar per_thread{seed, simdrng::thread{static_cast<std::uint64_t>(omp_get_thread_num())}};
simdrng::XoshiroScalar plain{seed};   // node, gpu and thread all at 0
```

The level constructor `Gen{seed, levels...}` and the `Gen{seed}.stream(levels...)`
member address the same stream.

A level tuple is a jump distance in the one sequence the seed gives.
The distance is `node * 2^a + gpu * 2^b + thread * 2^c`.
The bit fields are disjoint (`a > b > c`). One seed gives disjoint
streams. The levels do not touch the seed or the key. Each stream owns a block of `2^c` outputs, or
counter blocks for the counter-based generators. The table lists the field
widths and the stream length for each generator. The width table in
`include/simdrng/stream_levels.hpp` is the one source of truth.

| Generator         | `node` max | `gpu` max | `thread` max | Stream length                      |
|-------------------|------------|-----------|--------------|------------------------------------|
| Xoshiro256++      | no limit   | 2^32 - 1  | 2^32 - 1     | 2^127 outputs                      |
| SplitMix64        | 2^8 - 1    | 2^8 - 1   | 2^24 - 1     | 2^24 outputs                       |
| MWC64X32          | 2^8 - 1    | 2^8 - 1   | 2^24 - 1     | 2^22 outputs                       |
| ChaCha 8/12/20    | 2^16 - 1   | 2^16 - 1  | 2^32 - 1     | 2^64 counter blocks (2^67 outputs) |
| Philox 2x32       | 2^8 - 1    | 2^8 - 1   | 2^24 - 1     | 2^24 counter blocks                |
| Philox 4x32, 2x64 | 2^16 - 1   | 2^16 - 1  | 2^32 - 1     | 2^64 counter blocks                |
| Philox 4x64       | no limit   | no limit  | no limit     | 2^64 counter blocks                |

### SIMD lane layout

The Xoshiro256++ row covers `XoshiroScalar`, `XoshiroSIMD` and
`XoshiroNative`. All three accept the same tags. A tag tuple selects one
block of 2^127 outputs. A tagged `XoshiroSIMD` or `XoshiroNative` splits
that block between its lanes. Lane 0 starts at the block start and equals
`XoshiroScalar` with the same seed and tags. Lane i starts at `i * S` into
the block, with `S = 2^(127 - ceil(log2(W)))` for W lanes. Each lane owns S
outputs. The formula in `detail::simd_lane_stride_exp`
(`include/simdrng/xoshiro_simd.hpp`) is the one source of truth for S.

| W lanes      | S for each lane |
|--------------|-----------------|
| 2            | 2^126           |
| 4            | 2^125           |
| 8            | 2^124           |

The plain seed constructor `Gen{seed}` does not follow the tag layout. It
keeps the older layout with lane i at `i * 2^128`. Those lanes overlap
scalar `thread` streams of the same seed, so a program that mixes the two
layouts applies tags to each generator.

A stream that draws more than its length runs into the next stream's numbers,
and nothing checks for this. The rule holds for each generator with its own
length:

- A Xoshiro256++ stream that draws more than 2^127 outputs runs into the next
  stream's numbers, and nothing checks for this.
- A SplitMix64 stream that draws more than 2^24 outputs runs into the next
  stream's numbers, and nothing checks for this.
- A MWC64X32 stream that draws more than 2^22 outputs runs into the next
  stream's numbers, and nothing checks for this.
- A ChaCha 8/12/20 stream that draws more than 2^64 counter blocks repeats its
  own numbers, and nothing checks for this. The counter does not carry into the
  nonce, so the streams of one key stay disjoint.
- A Philox 2x32 stream that draws more than 2^24 counter blocks runs into the
  next stream's numbers, and nothing checks for this.
- A Philox 4x32 or 2x64 stream that draws more than 2^64 counter blocks runs
  into the next stream's numbers, and nothing checks for this.
- A Philox 4x64 stream that draws more than 2^64 counter blocks runs into the
  next stream's numbers, and nothing checks for this.

A level value above the max stops the program.
In a constant expression it is a compile error. On the host the program calls
`std::abort()`. In CUDA device code it calls `__trap()`.

On `XoshiroScalar`, a nonzero level in `stream()` or the level constructor
exceeds the constexpr step limit of clang and MSVC (1048576). Such calls work
at runtime only.

For the counter-based generators one counter block is one Philox block (2 or
4 words) or one 64-byte ChaCha block. The low counter words keep the
position in the stream block. Keep the position of the plain generator
below the stream length, so the blocks stay disjoint. Philox gives each work
item its own sub-stream from `(seed, counter)`, with no
coordination. See
[`examples/cpp/philox_counter_based.cpp`](examples/cpp/philox_counter_based.cpp)
and [`examples/cpp/threaded_openmp.cpp`](examples/cpp/threaded_openmp.cpp).

## GPU streams

All scalar generators compile in CUDA device code: SplitMix, XoshiroScalar,
MWC64X32, Philox and ChaCha. The headers compile without
`--expt-relaxed-constexpr`. `macros.hpp` marks all scalar members
`__host__ __device__`. `macros.hpp` has `simdrng::array`, a
host-and-device array type, and `simdrng_bit_cast`. In nvcc,
`simdrng_bit_cast` uses the CCCL `std::bit_cast`.

### CUDA test

Install the CUDA toolkit first. `SIMDRNG_CUDA` is OFF by default. Set it to
ON to make the CUDA device test:

```sh
cmake -S . -B build -DSIMDRNG_CUDA=ON
cmake --build build
ctest --test-dir build -R test_cuda
```

CMake sets `CMAKE_CUDA_ARCHITECTURES` to the host GPU. Set a value to change
the target. `test_cuda` runs the check on the GPU: device output equals
host output for all scalar classes, including `stream()`.

### Per-thread streams

A CUDA kernel gives each thread its own disjoint stream:

```cpp
#include <simdrng/xoshiro_scalar.hpp>
#include <cuda_runtime.h>

__global__ void fill(std::uint64_t seed, std::uint64_t *out) {
  const auto tid = static_cast<std::uint64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  auto g = simdrng::XoshiroScalar{seed, simdrng::thread{tid}};
  out[tid] = g();
}
```

### Multi-node streams

Use the levels together to address a stream on one node, GPU and thread. The
MPI rank is `node`, the CUDA device is `gpu`, the OpenMP thread is `thread`.
Each level adds its jump distance to the one sequence the seed gives:

```cpp
const std::uint64_t rank = MPI_rank, dev = gpu_id, tid = omp_get_thread_num();
simdrng::XoshiroScalar rng{42u, simdrng::node{rank}, simdrng::gpu{dev},
                           simdrng::thread{tid}};
```

### Limits

Vigna sets a limit on the overlap of random streams
(<https://prng.di.unimi.it/#period>). For period P, n streams of length L
starting at random points overlap with probability at most n^2 * L / P.
MWC64X32 holds a 64-bit state with a period of about 2^63, and Vigna calls
such a state space too small for parallel use. It fits register-scarce
hardware such as GPUs. For bulk CPU fill, use XoshiroSIMD. Its SIMD batches
are faster than the scalar loop.

## Install

```sh
git clone https://github.com/DiamonDinoia/simdrng.git
cd simdrng
cmake --preset release
cmake --build build/release
ctest --preset test
```

`CMakePresets.json` ships ready-made profiles so you rarely need raw `-D` flags:
`release`, `debug`, `ci`, `bench`, `sanitizers`, `tsan`, `valgrind`,
`static-analysis`, `coverage`.

CMake options:

| Option                    | Default | Purpose                                                               |
|---------------------------|---------|-----------------------------------------------------------------------|
| `SIMDRNG_BUILD_TESTS`     | ON\*    | Build Catch2 tests and benchmarks (\*follows `BUILD_TESTING`)         |
| `SIMDRNG_BUILD_PYTHON`    | OFF     | Build the nanobind Python extension                                   |
| `SIMDRNG_BUILD_EXAMPLES`  | OFF     | Build C++ examples under `examples/cpp`                               |
| `SIMDRNG_BUILD_DOCS`      | OFF     | Generate Sphinx/Doxygen docs                                          |
| `SIMDRNG_MARCH_NATIVE`    | OFF     | Compile benchmarks with `-march=native`                               |
| `SIMDRNG_USE_SANITIZERS`  | OFF     | `ON` = ASan+UBSan, `TSAN` = ThreadSanitizer                           |

**Consume from another CMake project** — install and `find_package`:

```cmake
find_package(simdrng CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE simdrng::simdrng)
```

The installed package config pulls in its dependencies (`xsimd`, `poet`) via
`find_dependency` automatically — but only when built with xsimd; a scalar-only
install requires no dependencies.

Or vendor it directly with FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(simdrng
    GIT_REPOSITORY https://github.com/DiamonDinoia/simdrng.git
    GIT_TAG        main)
FetchContent_MakeAvailable(simdrng)
target_link_libraries(my_target PRIVATE simdrng::simdrng)
```

### Single header

An amalgamated **`simdrng.hpp`** is published to the
[`single-header`](https://github.com/DiamonDinoia/simdrng/tree/single-header)
branch. poet is inlined; xsimd is kept as an external `<xsimd/...>` include, so
compile with the xsimd headers available. The compile-time-arch `*Native`
generators (`XoshiroNative`, `Philox4x64Native`, …) work header-only; the
runtime-dispatch types (`XoshiroSIMD`, `Philox*SIMD`, and the default
`simdrng::Xoshiro` alias) require linking the compiled library and are not
available from the single header alone.

### Python

```sh
pip install .
# or, for development:
pip install -e '.[test]' --no-build-isolation
```

## Benchmarks

The full matrix runs in CI and is rendered to SVG on the
[`benchmark-results`](https://github.com/DiamonDinoia/simdrng/tree/benchmark-results)
branch.

The chart below shows `u64` bulk throughput — the native output of every
engine — comparing the scalar loop against SIMD `generate()`. Per-call latency,
double generation (`uniform()` vs `std::uniform_real_distribution`), and memory
cost are covered in
[docs → benchmarks](https://simdrng.readthedocs.io/en/latest/benchmarks.html).

![Buffer-fill throughput, scalar vs SIMD (gcc-15)](https://raw.githubusercontent.com/DiamonDinoia/simdrng/benchmark-results/charts/gcc-15/throughput.svg)

## Documentation

Full documentation — quick start, per-family guides, the auto-generated C++ and
Python API reference, and the technical references — is hosted at
**<https://simdrng.readthedocs.io/>**.

## Contributing

Formatting and linting run through [pre-commit](https://pre-commit.com/)
(clang-format, gersemi, darker/ruff, shfmt, markdownlint). Install once:

```sh
pip install pre-commit && pre-commit install
git config blame.ignoreRevsFile .git-blame-ignore-revs   # skip the bulk-format commit in blame
```

## References

Grouped by the family each source backs — full annotations (periods, scrambling,
seeding, the `uniform()` rationale) live in
[docs → references](https://simdrng.readthedocs.io/en/latest/references.html):

- **Xoshiro256++ & SplitMix64** — D. Blackman and S. Vigna, *Scrambled Linear
  Pseudorandom Number Generators*, ACM TOMS (2021),
  <https://vigna.di.unimi.it/papers.php#BlVSLPNG>; and S. Vigna's
  xoshiro/xoroshiro shootout, SplitMix64 seeding, and `uniform()` rationale,
  <https://prng.di.unimi.it/>.
- **ChaCha 8/12/20** — D. J. Bernstein, *ChaCha, a variant of Salsa20*,
  <https://cr.yp.to/chacha.html>; cross-checked against
  [Monocypher](https://monocypher.org/) in the tests.
- **Philox** — J. K. Salmon et al., *Parallel Random Numbers: As Easy as
  1, 2, 3* (Philox), SC '11.

The reference `splitmix64.c` / `xoshiro256plusplus.c` used by the tests are the
authors' own public-domain (CC0) code from <https://prng.di.unimi.it/>.

## License

BSD-3-Clause-Attribution — see [`LICENSE`](LICENSE).
