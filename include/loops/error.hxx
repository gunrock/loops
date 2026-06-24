#pragma once

#include <exception>
#include <string>

#include <loops/util/xpu.hxx>

namespace loops {

/**
 * @namespace error
 * Error utilities for exception handling within device and host code.
 */
namespace error {

typedef xpu::error_t error_t;

/**
 * @brief Exception class for errors in device code.
 *
 */
struct exception_t : std::exception {
  std::string report;

  exception_t(error_t _status, std::string _message = "") {
    report = xpu::get_error_string(_status) + std::string("\t: ") + _message;
  }

  exception_t(std::string _message = "") { report = _message; }
  virtual const char* what() const noexcept { return report.c_str(); }
};

/**
 * @brief Throw an exception if the given error code is not a success.
 *
 * @param status error_t error code (xpu::success on no error).
 * @param message custom message to be appended to the error message.
 */
inline void throw_if_exception(error_t status, std::string message = "") {
  if (status != xpu::success)
    throw exception_t(status, message);
}

inline void throw_if_exception(bool is_exception, std::string message = "") {
  if (is_exception)
    throw exception_t(message);
}

}  // namespace error
}  // namespace loops
