// Probe for out-of-range stream levels. Run with one case name, it spawns
// itself for that case and requires the child to die hard (abort on the host)
// instead of aliasing another stream. The child hits the level value just past
// its index field; if it survives it prints "survived" and the parent fails
// the test. The child mode keeps the level values runtime-shaped (read from
// argv) so no constant evaluation can fold the call away.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <simdrng/mwc64x32.hpp>
#include <simdrng/philox.hpp>

namespace {

constexpr std::uint64_t kSeed = 0x0102030405060708ULL;

/// Runs the named case with the out-of-range level value; dies hard on the
/// guard. Returns only when the guard failed to fire.
std::uint64_t run_case(const char *name, std::uint64_t v) {
  if (std::strcmp(name, "philox2x32-gpu") == 0) {
    return simdrng::Philox2x32{kSeed}.stream(simdrng::gpu{v}, simdrng::thread{1})();
  }
  if (std::strcmp(name, "philox2x32-thread") == 0) {
    return simdrng::Philox2x32{kSeed}.stream(simdrng::gpu{1}, simdrng::thread{v})();
  }
  if (std::strcmp(name, "mwc64x32-gpu") == 0) {
    return simdrng::MWC64X32{kSeed}.stream(simdrng::gpu{v}, simdrng::thread{1})();
  }
  if (std::strcmp(name, "mwc64x32-thread") == 0) {
    return simdrng::MWC64X32{kSeed}.stream(simdrng::gpu{1}, simdrng::thread{v})();
  }
  std::printf("unknown case %s\n", name);
  std::exit(2);
}

} // namespace

int main(int argc, char **argv) {
  // Child mode: one level value just past the index field (gpu 8 bits for
  // Philox2x32, 7 for MWC64X32; thread 24 bits for both).
  if (argc == 4 && std::strcmp(argv[3], "run") == 0) {
    const std::uint64_t v = std::strtoull(argv[2], nullptr, 0);
    const std::uint64_t out = run_case(argv[1], v);
    std::printf("%s survived: 0x%016llx\n", argv[1], static_cast<unsigned long long>(out));
    return 0;
  }
  if (argc != 2) {
    return 2;
  }
  const char *v =
      (std::strstr(argv[1], "gpu") != nullptr) ? (std::strstr(argv[1], "mwc") != nullptr ? "128" : "256") : "16777216";
  const std::string cmd = std::string("\"") + argv[0] + "\" " + argv[1] + " " + v + " run";
  const int status = std::system(cmd.c_str());
  if (status == 0) {
    std::printf("%s: child survived, expected a hard failure\n", argv[1]);
    return 1;
  }
  std::printf("%s: child died as expected (status %d)\n", argv[1], status);
  return 0;
}
