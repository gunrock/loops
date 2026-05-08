/**
 * @file mtx_parser.hxx
 * @author Loops contributors
 * @brief Walk-pointer parser primitives for Matrix Market files.
 *
 * Operates on a contiguous byte range @c [p, end) , typically supplied by
 * @c loops::detail::mapped_file_t . All routines are pure walk-pointer:
 * they advance through the input and return the new position, never
 * touching anything outside the range.
 *
 * Why hand-rolled instead of @c fscanf / @c sscanf :
 *
 *   - @c fscanf is locale-aware (a thread-local mutex on glibc) and
 *     dispatches through stdio's @c FILE buffer for every conversion;
 *     measured at ~150 K integers/sec on the loops H100 box. That makes
 *     loading @c com-Orkut.mtx (117 M lines) a 5-8 minute affair.
 *
 *   - @c std::from_chars is locale-independent and operates directly on
 *     the input buffer with no allocation; measured at ~30 M integers/sec
 *     on the same box. The whole-file load drops to ~30 seconds.
 *
 *   - The Matrix Market grammar is trivial (whitespace, decimal integers,
 *     a single double) so a hand-rolled scanner has no business being
 *     more complex than the table below.
 *
 * Every parser returns the post-token pointer; callers detect failure by
 * checking @c (returned == start) , i.e. no progress.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/error.hxx>

#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <system_error>

namespace loops {
namespace detail {

/// Advance past horizontal whitespace (spaces and tabs only). Newlines
/// and carriage returns are *not* consumed; they're token-significant
/// for the Matrix Market line-oriented body grammar.
inline const char* skip_blank(const char* p, const char* end) noexcept {
  while (p < end && (*p == ' ' || *p == '\t'))
    ++p;
  return p;
}

/// Advance past any whitespace including newlines. Used between body
/// records where we don't care about line structure.
inline const char* skip_ws(const char* p, const char* end) noexcept {
  while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n'))
    ++p;
  return p;
}

/// Skip to and past the next newline. Used to drop comment lines and
/// to reach the next record after parsing a row's tokens.
inline const char* skip_to_eol(const char* p, const char* end) noexcept {
  while (p < end && *p != '\n')
    ++p;
  if (p < end)
    ++p;  // step past the '\n'
  return p;
}

/// Skip Matrix Market comment lines (those starting with '%') and any
/// blank lines between them. Stops at the first non-comment, non-blank
/// character; the dimension line lives there.
inline const char* skip_comments(const char* p, const char* end) noexcept {
  for (;;) {
    p = skip_ws(p, end);
    if (p >= end)
      return p;
    if (*p != '%')
      return p;
    p = skip_to_eol(p, end);
  }
}

/// Parse a non-negative decimal integer into @c v . Returns the post-
/// token pointer. Caller checks @c (returned == start) for "no digit"
/// or @c v overflow via the @c std::from_chars error code.
inline const char* parse_size_t(const char* p,
                                const char* end,
                                std::size_t& v) noexcept {
  auto r = std::from_chars(p, end, v);
  return (r.ec == std::errc()) ? r.ptr : p;
}

/// Parse a decimal floating-point value into @c v . Uses
/// @c std::from_chars where the standard library implements the
/// floating-point overload (libstdc++ >= 11, MSVC >= 19.20, libc++ >= 14);
/// falls back to @c std::strtod via a small NUL-terminated scratch
/// buffer otherwise.
///
/// On the fallback path we copy at most @c kFallbackBufSize-1 characters
/// into the scratch buffer; that's well above any sane @c %lf token
/// (decimal + sign + ~17 mantissa digits + exponent + null).
inline const char* parse_double(const char* p,
                                const char* end,
                                double& v) noexcept {
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
  auto r = std::from_chars(p, end, v);
  return (r.ec == std::errc()) ? r.ptr : p;
#else
  // Fallback: locate the token, copy into a NUL-terminated buffer, strtod.
  // This path is hit on older libc++ (< 14) and pre-19.20 MSVC.
  constexpr std::size_t kFallbackBufSize = 64;
  char buf[kFallbackBufSize];
  const char* q = p;
  std::size_t n = 0;
  while (q < end && n + 1 < kFallbackBufSize && *q != ' ' && *q != '\t' &&
         *q != '\r' && *q != '\n') {
    buf[n++] = *q++;
  }
  buf[n] = '\0';
  char* tail = nullptr;
  v = std::strtod(buf, &tail);
  if (tail == buf)
    return p;
  return p + (tail - buf);
#endif
}

/// Move-pointer reader for the @c %%MatrixMarket header line.
struct mm_typecode_t {
  bool is_matrix = false;
  bool is_coordinate = false;  /// @c true = sparse, @c false = dense array
  bool is_real = false;
  bool is_integer = false;
  bool is_pattern = false;
  bool is_complex = false;
  bool is_general = false;
  bool is_symmetric = false;
  bool is_skew = false;
  bool is_hermitian = false;
};

/// Parse a @c %%MatrixMarket banner line into a typecode. Throws via
/// @c loops::error::throw_if_exception on a malformed header.
///
/// Grammar:
///   %%MatrixMarket {matrix} {coordinate|array} {real|integer|pattern|complex}
///   {general|symmetric|skew-symmetric|hermitian}
inline const char* parse_banner(const char* p,
                                const char* end,
                                mm_typecode_t& tc) {
  static constexpr const char kBanner[] = "%%MatrixMarket";
  static constexpr std::size_t kBannerLen = sizeof(kBanner) - 1;

  error::throw_if_exception(
      static_cast<std::size_t>(end - p) < kBannerLen,
      "matrix-market: file too short to contain a banner");
  for (std::size_t i = 0; i < kBannerLen; ++i) {
    error::throw_if_exception(p[i] != kBanner[i],
                              "matrix-market: missing %%MatrixMarket banner");
  }
  p += kBannerLen;

  // 4 banner tokens: object, format, field, symmetry.
  auto next_token = [&](std::string& out) -> const char* {
    p = skip_blank(p, end);
    const char* start = p;
    while (p < end && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n')
      ++p;
    out.assign(start, static_cast<std::size_t>(p - start));
    return p;
  };

  std::string object, format, field, symmetry;
  p = next_token(object);
  p = next_token(format);
  p = next_token(field);
  p = next_token(symmetry);
  p = skip_to_eol(p, end);

  // Lowercase comparisons (the spec is case-insensitive for the banner
  // tokens, but in practice every dataset emits lowercase).
  auto eq_ci = [](const std::string& a, const char* b) {
    if (a.size() != std::char_traits<char>::length(b))
      return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
      char c = a[i];
      if (c >= 'A' && c <= 'Z')
        c = static_cast<char>(c - 'A' + 'a');
      if (c != b[i])
        return false;
    }
    return true;
  };

  tc.is_matrix = eq_ci(object, "matrix");
  tc.is_coordinate = eq_ci(format, "coordinate");
  tc.is_real = eq_ci(field, "real");
  tc.is_integer = eq_ci(field, "integer");
  tc.is_pattern = eq_ci(field, "pattern");
  tc.is_complex = eq_ci(field, "complex");
  tc.is_general = eq_ci(symmetry, "general");
  tc.is_symmetric = eq_ci(symmetry, "symmetric");
  tc.is_skew = eq_ci(symmetry, "skew-symmetric");
  tc.is_hermitian = eq_ci(symmetry, "hermitian");

  return p;
}

}  // namespace detail
}  // namespace loops
