/**
 * @file parameters.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief SpMV + NVBench parameters.
 * @version 0.1
 * @date 2022-07-18
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <string>
#include <iostream>
#include <cxxopts.hpp>

#include <loops/util/filepath.hxx>
#include <nvbench/nvbench.cuh>

std::string filename;

struct parameters_t {
  /**
   * @brief Construct a new parameters object and parse command line arguments.
   *
   * @param argc Number of command line arguments.
   * @param argv Command line arguments.
   */
  parameters_t(int argc, char** argv)
      : m_options(argv[0], "SPMV Benchmarking"), m_argc(argc) {
    m_options.allow_unrecognised_options();
    // Add command line options
    m_options.add_options()("h,help", "Print help")  // help
        ("m,market", "Matrix file",
         cxxopts::value<std::string>());  // mtx

    // Parse command line arguments.
    auto result = m_options.parse(argc, argv);

    // Print help if requested
    if (result.count("help")) {
      m_help = true;
      std::cout << m_options.help({""});
      std::cout << "  [optional nvbench args]" << std::endl << std::endl;
      const char* argh[1] = {"-h"};
      NVBENCH_MAIN_BODY(1, argh);
    }

    // Get matrix market file or error if not specified.
    else {
      if (result.count("market") == 1) {
        this->m_filename = result["market"].as<std::string>();
        filename = m_filename;
        if (!loops::is_market(m_filename)) {
          std::cout << m_options.help({""});
          std::cout << "  [optional nvbench args]" << std::endl << std::endl;
          std::exit(0);
        }

        // Remove loops parameters and pass the rest to nvbench.
        for (int i = 0; i < argc; i++) {
          if (strcmp(argv[i], "--market") == 0 || strcmp(argv[i], "-m") == 0) {
            i++;
            continue;
          }
          m_args.push_back(argv[i]);
        }

      } else {
        std::cout << m_options.help({""});
        std::cout << "  [optional nvbench args]" << std::endl << std::endl;
        std::exit(0);
      }
    }
  }

  /// Helpers for NVBENCH_MAIN_BODY call.
  int nvbench_argc() { return m_argc - 2; }
  auto nvbench_argv() { return m_args.data(); }

 private:
  std::string m_filename;
  cxxopts::Options m_options;
  std::vector<const char*> m_args;
  bool m_help = false;
  int m_argc;
};