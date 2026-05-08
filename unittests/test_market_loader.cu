/**
 * @file test_market_loader.cu
 * @author Loops contributors
 * @brief Unit tests for the mmap + std::from_chars Matrix Market loader.
 *
 * Every test writes a small @c .mtx file into a unique scratch path under
 * the system temp directory, runs @c matrix_market_t::load , and asserts
 * on the resulting host @c coo_t . The temp file is removed on success
 * and on Catch2's @c CHECK -triggered teardown (RAII).
 *
 * Coverage:
 *   - General real matrix (no symmetry expansion).
 *   - Integer matrix (decimal weights parsed as @c type_t ).
 *   - Pattern matrix (no value column; entries default to 1).
 *   - Symmetric real matrix (off-diagonal mirroring; diagonal kept once).
 *   - Symmetric pattern matrix (no value column AND mirror).
 *   - Body parser tolerates leading / trailing / mid-stream comment lines.
 *   - Banner with skewed-symmetric / hermitian / complex / array fails fast.
 *   - Zero-indexed entries fail fast (Matrix Market is 1-indexed).
 *
 * @copyright Copyright (c) 2026
 *
 */

#include <catch2/catch_test_macros.hpp>

#include <loops/container/coo.hxx>
#include <loops/container/market.hxx>
#include <loops/error.hxx>
#include <loops/memory.hxx>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace loops;
namespace fs = std::filesystem;

namespace {

/// RAII scratch file under the OS temp dir. The body is written verbatim;
/// the destructor unlinks the file even if the test asserts mid-way.
class temp_mtx_t {
 public:
  explicit temp_mtx_t(const std::string& body) {
    static std::atomic<unsigned> counter{0};
    // Combine a per-process steady-clock stamp with an atomic counter so
    // tests in the same binary can run in parallel without colliding on
    // path names (and without depending on a POSIX getpid()).
    const auto stamp = static_cast<unsigned long long>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto tag = counter.fetch_add(1, std::memory_order_relaxed);
    path_ = fs::temp_directory_path() /
            ("loops_mm_test_" + std::to_string(stamp) + "_" +
             std::to_string(tag) + ".mtx");
    std::ofstream f(path_, std::ios::binary | std::ios::trunc);
    REQUIRE(f.is_open());
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
    REQUIRE(static_cast<bool>(f));
  }

  ~temp_mtx_t() {
    std::error_code ec;
    fs::remove(path_, ec);  // best-effort
  }

  temp_mtx_t(const temp_mtx_t&) = delete;
  temp_mtx_t& operator=(const temp_mtx_t&) = delete;

  std::string str() const { return path_.string(); }

 private:
  fs::path path_;
};

/// Look up @c (r, c) in a host COO and return the matching value, if any.
/// Linear search is fine for the tiny matrices these tests use.
template <typename Coo>
bool find_entry(const Coo& coo, int r, int c, float& v_out) {
  for (std::size_t i = 0; i < coo.nnzs; ++i) {
    if (static_cast<int>(coo.row_indices[i]) == r &&
        static_cast<int>(coo.col_indices[i]) == c) {
      v_out = static_cast<float>(coo.values[i]);
      return true;
    }
  }
  return false;
}

}  // namespace

TEST_CASE("matrix_market_t loads a general real matrix",
          "[market][general][real]") {
  // 3x3 dense-ish:
  //   1.5  0    0
  //   0    2.0  3.0
  //   4.0  0    0
  temp_mtx_t mtx(
      "%%MatrixMarket matrix coordinate real general\n"
      "% example general real matrix\n"
      "3 3 4\n"
      "1 1 1.5\n"
      "2 2 2.0\n"
      "2 3 3.0\n"
      "3 1 4.0\n");

  matrix_market_t<int, int, float> reader;
  auto coo = reader.load(mtx.str());

  CHECK(coo.rows == 3);
  CHECK(coo.cols == 3);
  CHECK(coo.nnzs == 4);

  float v;
  REQUIRE(find_entry(coo, 0, 0, v));
  CHECK(v == 1.5f);
  REQUIRE(find_entry(coo, 1, 1, v));
  CHECK(v == 2.0f);
  REQUIRE(find_entry(coo, 1, 2, v));
  CHECK(v == 3.0f);
  REQUIRE(find_entry(coo, 2, 0, v));
  CHECK(v == 4.0f);
}

TEST_CASE("matrix_market_t loads an integer matrix as the value type",
          "[market][general][integer]") {
  temp_mtx_t mtx(
      "%%MatrixMarket matrix coordinate integer general\n"
      "2 2 2\n"
      "1 1 7\n"
      "2 2 -3\n");

  matrix_market_t<int, int, float> reader;
  auto coo = reader.load(mtx.str());

  CHECK(coo.nnzs == 2);
  float v;
  REQUIRE(find_entry(coo, 0, 0, v));
  CHECK(v == 7.0f);
  REQUIRE(find_entry(coo, 1, 1, v));
  CHECK(v == -3.0f);
}

TEST_CASE("matrix_market_t pattern matrices default values to 1",
          "[market][general][pattern]") {
  temp_mtx_t mtx(
      "%%MatrixMarket matrix coordinate pattern general\n"
      "3 3 3\n"
      "1 2\n"
      "2 3\n"
      "3 1\n");

  matrix_market_t<int, int, float> reader;
  auto coo = reader.load(mtx.str());

  CHECK(coo.nnzs == 3);
  for (std::size_t i = 0; i < coo.nnzs; ++i) {
    CHECK(coo.values[i] == 1.0f);
  }
}

TEST_CASE("matrix_market_t expands symmetric off-diagonals",
          "[market][symmetric][real]") {
  // Symmetric 3x3 (lower triangle only on disk):
  //   1    2    .
  //   2    3    .
  //   .    .    4
  // Expanded form has 5 entries (diagonal kept once, off-diagonal mirrored).
  temp_mtx_t mtx(
      "%%MatrixMarket matrix coordinate real symmetric\n"
      "% header comment\n"
      "3 3 3\n"
      "1 1 1.0\n"
      "2 1 2.0\n"
      "3 3 4.0\n"
      "2 2 3.0\n");

  matrix_market_t<int, int, float> reader;
  auto coo = reader.load(mtx.str());

  CHECK(coo.rows == 3);
  CHECK(coo.cols == 3);
  CHECK(coo.nnzs == 5);  // 3 records + 1 off-diagonal mirror

  float v;
  REQUIRE(find_entry(coo, 0, 0, v));
  CHECK(v == 1.0f);
  REQUIRE(find_entry(coo, 1, 1, v));
  CHECK(v == 3.0f);
  REQUIRE(find_entry(coo, 2, 2, v));
  CHECK(v == 4.0f);
  REQUIRE(find_entry(coo, 1, 0, v));
  CHECK(v == 2.0f);
  REQUIRE(find_entry(coo, 0, 1, v));  // mirror
  CHECK(v == 2.0f);
}

TEST_CASE("matrix_market_t handles symmetric pattern matrices",
          "[market][symmetric][pattern]") {
  // Undirected graph: edges (0-1), (1-2). Symmetric pattern.
  temp_mtx_t mtx(
      "%%MatrixMarket matrix coordinate pattern symmetric\n"
      "3 3 2\n"
      "2 1\n"
      "3 2\n");

  matrix_market_t<int, int, float> reader;
  auto coo = reader.load(mtx.str());

  CHECK(coo.nnzs == 4);  // 2 edges, mirrored.
  float v;
  REQUIRE(find_entry(coo, 0, 1, v));
  CHECK(v == 1.0f);
  REQUIRE(find_entry(coo, 1, 0, v));
  CHECK(v == 1.0f);
  REQUIRE(find_entry(coo, 1, 2, v));
  CHECK(v == 1.0f);
  REQUIRE(find_entry(coo, 2, 1, v));
  CHECK(v == 1.0f);
}

TEST_CASE("matrix_market_t skips comment lines anywhere in the header",
          "[market][parser][comments]") {
  // Comments before, between, and after the dim line.
  temp_mtx_t mtx(
      "%%MatrixMarket matrix coordinate real general\n"
      "% top comment\n"
      "%\n"
      "% another\n"
      "2 2 2\n"
      "1 1 1.0\n"
      "2 2 2.0\n");

  matrix_market_t<int, int, float> reader;
  auto coo = reader.load(mtx.str());

  CHECK(coo.nnzs == 2);
}

TEST_CASE("matrix_market_t rejects unsupported variants",
          "[market][parser][errors]") {
  SECTION("complex") {
    temp_mtx_t mtx(
        "%%MatrixMarket matrix coordinate complex general\n"
        "1 1 1\n"
        "1 1 1.0 0.0\n");
    matrix_market_t<int, int, float> reader;
    CHECK_THROWS(reader.load(mtx.str()));
  }
  SECTION("hermitian") {
    temp_mtx_t mtx(
        "%%MatrixMarket matrix coordinate complex hermitian\n"
        "1 1 1\n"
        "1 1 1.0 0.0\n");
    matrix_market_t<int, int, float> reader;
    CHECK_THROWS(reader.load(mtx.str()));
  }
  SECTION("skew-symmetric") {
    temp_mtx_t mtx(
        "%%MatrixMarket matrix coordinate real skew-symmetric\n"
        "2 2 1\n"
        "2 1 1.0\n");
    matrix_market_t<int, int, float> reader;
    CHECK_THROWS(reader.load(mtx.str()));
  }
  SECTION("dense array") {
    temp_mtx_t mtx(
        "%%MatrixMarket matrix array real general\n"
        "1 1\n"
        "1.0\n");
    matrix_market_t<int, int, float> reader;
    CHECK_THROWS(reader.load(mtx.str()));
  }
  SECTION("missing banner") {
    temp_mtx_t mtx(
        "% no banner above me\n"
        "1 1 1\n"
        "1 1 1.0\n");
    matrix_market_t<int, int, float> reader;
    CHECK_THROWS(reader.load(mtx.str()));
  }
  SECTION("zero-indexed") {
    temp_mtx_t mtx(
        "%%MatrixMarket matrix coordinate real general\n"
        "2 2 1\n"
        "0 1 1.0\n");
    matrix_market_t<int, int, float> reader;
    CHECK_THROWS(reader.load(mtx.str()));
  }
}

TEST_CASE("matrix_market_t propagates type_t through symmetric expansion",
          "[market][symmetric][f64]") {
  // The same off-diagonal mirror used to copy the value through an
  // intermediate vector_t<type_t>; verify the f64 path round-trips.
  temp_mtx_t mtx(
      "%%MatrixMarket matrix coordinate real symmetric\n"
      "2 2 2\n"
      "1 1 1.5\n"
      "2 1 2.5\n");

  matrix_market_t<int, int, double> reader;
  auto coo = reader.load(mtx.str());
  CHECK(coo.nnzs == 3);

  bool saw_mirror = false;
  for (std::size_t i = 0; i < coo.nnzs; ++i) {
    if (coo.row_indices[i] == 0 && coo.col_indices[i] == 1) {
      CHECK(coo.values[i] == 2.5);
      saw_mirror = true;
    }
  }
  CHECK(saw_mirror);
}
