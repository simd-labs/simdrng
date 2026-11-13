// Probe for out-of-range stream levels. Run with one case name, it spawns
// itself for that case and requires the child to die hard (abort on the host)
// instead of aliasing another stream. The child hits the level value just past
// its field; if it survives it prints "survived" and the parent fails the
// test. The child mode keeps the level values runtime-shaped (read from
// argv) so no constant evaluation can fold the call away. Case names are
// "<generator>-<level>"; the out-of-range value is 1 << field_bits, taken
// from the width table in <simdrng/stream_levels.hpp>. A 64-bit field holds
// every std::uint64_t and gets no case.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <simdrng/chacha.hpp>
#include <simdrng/mwc64x32.hpp>
#include <simdrng/philox.hpp>
#include <simdrng/splitmix.hpp>
#include <simdrng/xoshiro_scalar.hpp>
#include <simdrng/xoshiro_simd.hpp>

namespace {

constexpr std::uint64_t kSeed = 0x0102030405060708ULL;
constexpr std::array<std::uint32_t, 8> kChaChaKey = {0x01020304, 0x05060708, 0x09101112, 0x13141516,
                                                     0x17181920, 0x21222324, 0x25262728, 0x29303132};

/// The width-table row of the named generator.
simdrng::detail::stream_widths widths_for(const char *name) {
  if (std::strstr(name, "xoshiro") != nullptr) {
    return simdrng::detail::widths_xoshiro256;
  }
  if (std::strstr(name, "philox4x64") != nullptr) {
    return simdrng::detail::widths_counter256;
  }
  if (std::strstr(name, "philox4x32") != nullptr || std::strstr(name, "philox2x64") != nullptr ||
      std::strstr(name, "chacha") != nullptr) {
    return simdrng::detail::widths_counter128;
  }
  return simdrng::detail::widths_tight; // splitmix, mwc64x32, philox2x32
}

/// The first level value past the named field.
std::uint64_t oob_value(const char *name) {
  const auto w = widths_for(name);
  const std::uint64_t bits = std::strstr(name, "-node") != nullptr  ? w.node_bits
                             : std::strstr(name, "-gpu") != nullptr ? w.gpu_bits
                                                                    : w.thread_bits;
  return std::uint64_t{1} << bits;
}

template <class G> std::uint64_t draw_leveled(const G &base, const std::string &level, std::uint64_t v) {
  if (level == "node") {
    return base.stream(simdrng::node{v})();
  }
  if (level == "gpu") {
    return base.stream(simdrng::gpu{v})();
  }
  return base.stream(simdrng::thread{v})();
}

/// Runs the named case with the out-of-range level value; dies hard on the
/// guard. Returns only when the guard failed to fire.
std::uint64_t run_case(const char *name, std::uint64_t v) {
  const std::string full(name);
  const auto dash = full.find('-');
  const std::string gen = full.substr(0, dash);
  const std::string level = full.substr(dash + 1);
  if (gen == "splitmix") {
    return draw_leveled(simdrng::SplitMix{kSeed}, level, v);
  }
  if (gen == "mwc64x32") {
    return draw_leveled(simdrng::MWC64X32{kSeed}, level, v);
  }
  if (gen == "philox2x32") {
    return draw_leveled(simdrng::Philox2x32{kSeed}, level, v);
  }
  if (gen == "philox4x32") {
    return draw_leveled(simdrng::Philox4x32{kSeed}, level, v);
  }
  if (gen == "philox2x64") {
    return draw_leveled(simdrng::Philox2x64{kSeed}, level, v);
  }
  if (gen == "xoshiro") {
    return draw_leveled(simdrng::XoshiroScalar{kSeed}, level, v);
  }
  if (gen == "xoshirosimd") {
    return draw_leveled(simdrng::XoshiroSIMD{kSeed}, level, v);
  }
#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE
  if (gen == "xoshironative") {
    return draw_leveled(simdrng::XoshiroNative{kSeed}, level, v);
  }
#endif
  if (gen == "chacha") {
    return draw_leveled(simdrng::ChaCha<>{kChaChaKey, 0, 0}, level, v);
  }
  std::printf("unknown case %s\n", name);
  std::exit(2);
}

} // namespace

int main(int argc, char **argv) {
  // Child mode: one level value past the field (oob_value).
  if (argc == 4 && std::strcmp(argv[3], "run") == 0) {
    const std::uint64_t v = std::strtoull(argv[2], nullptr, 0);
    const std::uint64_t out = run_case(argv[1], v);
    std::printf("%s survived: 0x%016llx\n", argv[1], static_cast<unsigned long long>(out));
    return 0;
  }
  if (argc != 2) {
    return 2;
  }
  const std::string cmd =
      std::string("\"") + argv[0] + "\" " + argv[1] + " " + std::to_string(oob_value(argv[1])) + " run";
  const int status = std::system(cmd.c_str());
  if (status == 0) {
    std::printf("%s: child survived, expected a hard failure\n", argv[1]);
    return 1;
  }
  std::printf("%s: child died as expected (status %d)\n", argv[1], status);
  return 0;
}
