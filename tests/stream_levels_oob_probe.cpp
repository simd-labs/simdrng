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

#include <simdrng/chacha.hpp>
#include <simdrng/mwc64x32.hpp>
#include <simdrng/philox.hpp>
#include <simdrng/splitmix.hpp>

namespace {

constexpr std::uint64_t kSeed = 0x0102030405060708ULL;
constexpr std::array<std::uint32_t, 8> kChaChaKey = {0x01020304, 0x05060708, 0x09101112, 0x13141516,
                                                     0x17181920, 0x21222324, 0x25262728, 0x29303132};

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
  if (std::strcmp(name, "splitmix-gpu") == 0) {
    return simdrng::SplitMix{kSeed}.stream(simdrng::gpu{v}, simdrng::thread{1})();
  }
  if (std::strcmp(name, "splitmix-thread") == 0) {
    return simdrng::SplitMix{kSeed}.stream(simdrng::gpu{1}, simdrng::thread{v})();
  }
  if (std::strcmp(name, "chacha-gpu") == 0) {
    return simdrng::ChaCha<>{kChaChaKey, 0, 0}.stream(simdrng::gpu{v}, simdrng::thread{1})();
  }
  if (std::strcmp(name, "chacha-thread") == 0) {
    return simdrng::ChaCha<>{kChaChaKey, 0, 0}.stream(simdrng::gpu{1}, simdrng::thread{v})();
  }
  std::printf("unknown case %s\n", name);
  std::exit(2);
}

/// The first level value after the packed field of the named case.
const char *oob_value(const char *name) {
  if (std::strstr(name, "chacha") != nullptr) {
    return "4294967296"; // both nonce halves are 32 bits wide
  }
  if (std::strstr(name, "gpu") != nullptr) {
    // gpu: 7 bits for MWC64X32, 8 bits for Philox2x32 and SplitMix.
    return std::strstr(name, "mwc") != nullptr ? "128" : "256";
  }
  return "16777216"; // thread: 24 bits
}

} // namespace

int main(int argc, char **argv) {
  // Child mode: one level value past the index field (oob_value).
  if (argc == 4 && std::strcmp(argv[3], "run") == 0) {
    const std::uint64_t v = std::strtoull(argv[2], nullptr, 0);
    const std::uint64_t out = run_case(argv[1], v);
    std::printf("%s survived: 0x%016llx\n", argv[1], static_cast<unsigned long long>(out));
    return 0;
  }
  if (argc != 2) {
    return 2;
  }
  const std::string cmd = std::string("\"") + argv[0] + "\" " + argv[1] + " " + oob_value(argv[1]) + " run";
  const int status = std::system(cmd.c_str());
  if (status == 0) {
    std::printf("%s: child survived, expected a hard failure\n", argv[1]);
    return 1;
  }
  std::printf("%s: child died as expected (status %d)\n", argv[1], status);
  return 0;
}
