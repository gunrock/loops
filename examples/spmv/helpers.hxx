/**
 * @file helpers.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Header file for SpMV.
 * @version 0.1
 * @date 2022-02-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

/// Compile-time element type for the SpMV examples.
///
/// CMake builds @c .f32 (default) and @c .f64 variants of every example
/// by injecting @c -DLOOPS_VALUE_T=float or @c -DLOOPS_VALUE_T=double .
/// Sources read @c using @c type_t = LOOPS_VALUE_T; , so adding a new
/// precision is one CMake target + one macro define -- no per-example
/// edits.
#ifndef LOOPS_VALUE_T
#define LOOPS_VALUE_T float
#endif

#include <loops/util/generate.hxx>
#include <loops/util/reference.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/container/market.hxx>
#include <loops/util/filepath.hxx>
#include <loops/util/equal.hxx>
#include <loops/util/device.hxx>
#include <loops/util/timer.hxx>
#include <loops/memory.hxx>
#include <cxxopts.hpp>

#include <algorithm>
#include <iostream>

struct parameters_t {
  std::string filename;
  bool validate;
  bool verbose;
  bool rigorous;
  cxxopts::Options options;

  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv)
      : options(argv[0], "Sparse Matrix-Vector Multiplication") {
    options.add_options()("h,help", "Print help")                   // help
        ("m,market", "Matrix file", cxxopts::value<std::string>())  // mtx
        ("validate", "CPU validation")                              // validate
        ("rigorous",
         "Rigorous validation: f64 reference + Wilkinson per-row bound")  //
        ("v,verbose", "Verbose output");                                  //

    auto result = options.parse(argc, argv);

    if (result.count("help") || (result.count("market") == 0)) {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    if (result.count("market") == 1) {
      filename = result["market"].as<std::string>();
      if (loops::is_market(filename)) {
      } else {
        std::cout << options.help({""}) << std::endl;
        std::exit(0);
      }
    } else {
      std::cout << options.help({""}) << std::endl;
      std::exit(0);
    }

    validate = result.count("validate") == 1;
    rigorous = result.count("rigorous") == 1;
    verbose = result.count("verbose") == 1;
    if (rigorous) validate = true;  // rigorous implies validate
  }
};

namespace cpu {

using namespace loops;
using namespace loops::memory;

/**
 * @brief Example-side validation: compute the host CSR reference, count
 * mismatches against the GPU result, and print a one-line summary.
 *
 * Both the reference computation and the mismatch counting live in
 * @c loops::reference (see @c include/loops/util/reference.hxx ); this
 * thin wrapper exists so the example binaries can keep the pretty
 * "Matrix / Dimensions / Errors" stdout block the test harness scrapes.
 *
 * When @c parameters.rigorous is set, additionally runs
 * @c rigorously_validate_spmv (double-precision reference + per-row
 * Wilkinson bound) and reports a @c VERDICT line:
 *   - @c NOT_A_BUG if no row exceeds the per-row floating-point bound
 *     against the f64 reference (the @e naive errors are then just
 *     tolerance noise).
 *   - @c POTENTIAL_BUG otherwise.
 */
template <typename index_t, typename offset_t, typename type_t>
void validate(parameters_t& parameters,
              csr_t<index_t, offset_t, type_t>& csr,
              vector_t<type_t>& x,
              vector_t<type_t>& y) {
  auto h_y = loops::reference::spmv(csr, x);
  std::size_t errors = loops::reference::count_errors(
      y.data().get(), h_y.data(), csr.rows, parameters.verbose);

  std::cout << "Matrix:\t\t" << extract_filename(parameters.filename)
            << std::endl;
  std::cout << "Dimensions:\t" << csr.rows << " x " << csr.cols << " ("
            << csr.nnzs << ")" << std::endl;
  std::cout << "Errors:\t\t" << errors << std::endl;

  if (!parameters.rigorous) return;

  auto report = loops::reference::rigorously_validate_spmv(
      csr, x, y.data().get(),
      /*wilkinson_k=*/8.0, /*atol_floor=*/1e-3,
      parameters.verbose);

  std::cout << "WilkinsonK:\t" << report.wilkinson_k << std::endl;
  std::cout << "NaiveMismatches:\t" << report.naive_mismatches << std::endl;
  std::cout << "F32BaselineOverruns:\t" << report.f32_baseline_overruns
            << std::endl;
  std::cout << "GPUOverruns:\t" << report.gpu_overruns << std::endl;
  std::cout << "MaxAbsError:\t" << report.max_gpu_abs_error << std::endl;
  std::cout << "MaxRelError:\t" << report.max_gpu_rel_error << std::endl;
  std::cout << "Verdict:\t"
            << (report.gpu_overruns == 0 ? "NOT_A_BUG" : "POTENTIAL_BUG")
            << std::endl;
}

}  // namespace cpu