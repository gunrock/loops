/**
 * @file timer.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Simple timer utility for device side code.
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <loops/backend/xpu.hxx>

namespace loops {
namespace util {

struct timer_t {
  float time;

  timer_t() {
    xpu::event_create(&start_);
    xpu::event_create(&stop_);
    xpu::event_record(start_);
  }

  ~timer_t() {
    xpu::event_destroy(start_);
    xpu::event_destroy(stop_);
  }

  // Alias of each other, start the timer.
  void begin() { xpu::event_record(start_); }
  void start() { this->begin(); }

  // Alias of each other, stop the timer.
  float end() {
    xpu::event_record(stop_);
    xpu::event_synchronize(stop_);
    xpu::event_elapsed_time(&time, start_, stop_);

    return milliseconds();
  }
  float stop() { return this->end(); }

  float seconds() { return time * 1e-3; }
  float milliseconds() { return time; }

 private:
  xpu::event_t start_, stop_;
};

}  // namespace util
}  // namespace loops