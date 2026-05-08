/**
 * @file mapped_file.hxx
 * @author Loops contributors
 * @brief Portable read-only memory map of an entire file.
 *
 * The Matrix Market loader has to chew through 100M+ line files for the
 * SuiteSparse SNAP / LAW graphs. @c fscanf -driven parsing is the wrong tool
 * for that: it's serializing every byte through stdio's locale-aware lexer
 * and a 1 KiB userspace buffer, which on @c com-Orkut.mtx works out to 5-8
 * minutes of CPU just on tokenization.
 *
 * The right tool is to map the whole file into the process address space
 * once and let the parser walk the bytes directly. This trades nothing
 * (the OS pages on demand, so peak RSS isn't materially higher than
 * fscanf's stdio buffer) for a ~10x parse speed-up via @c std::from_chars.
 *
 * Usage:
 * @code
 *   loops::detail::mapped_file_t f("path/to/big.mtx");
 *   const char* p   = f.data();
 *   const char* end = p + f.size();
 *   // ... walk-pointer parser over [p, end) ...
 * @endcode
 *
 * The mapping is read-only and shared with the page cache, which means
 * two binaries loading the same matrix back-to-back (as the sweep does:
 * f32 then f64) hit the cache on the second load instead of re-reading
 * the whole file from disk.
 *
 * @copyright Copyright (c) 2026
 *
 */

#pragma once

#include <loops/error.hxx>

#include <cstddef>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h pollutes the global namespace with min / max macros even
// under NOMINMAX in some toolchains; nuke them so std::min / std::max
// in downstream code doesn't expand into garbage.
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace loops {
namespace detail {

/**
 * @brief RAII read-only memory map of a file.
 *
 * Empty files map to @c data() == @c nullptr / @c size() == @c 0 ; that's
 * a valid-but-empty mapping, not an error. Failure to open or map throws
 * via @c loops::error::throw_if_exception .
 *
 * Move-only: the destructor is responsible for unmapping and closing
 * the OS handles, so we forbid copies.
 */
class mapped_file_t {
 public:
  explicit mapped_file_t(const std::string& path) {
#if defined(_WIN32)
    file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    error::throw_if_exception(file_ == INVALID_HANDLE_VALUE,
                              "mapped_file_t: could not open " + path);

    LARGE_INTEGER sz;
    error::throw_if_exception(GetFileSizeEx(file_, &sz) == 0,
                              "mapped_file_t: could not query size of " + path);
    size_ = static_cast<std::size_t>(sz.QuadPart);

    if (size_ > 0) {
      mapping_ =
          CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
      error::throw_if_exception(
          mapping_ == nullptr,
          "mapped_file_t: CreateFileMapping failed for " + path);
      data_ = static_cast<const char*>(
          MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
      error::throw_if_exception(
          data_ == nullptr, "mapped_file_t: MapViewOfFile failed for " + path);
    }
#else
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    error::throw_if_exception(fd_ < 0, "mapped_file_t: could not open " + path);

    struct stat st;
    error::throw_if_exception(::fstat(fd_, &st) != 0,
                              "mapped_file_t: fstat failed on " + path);
    size_ = static_cast<std::size_t>(st.st_size);

    if (size_ > 0) {
      void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
      error::throw_if_exception(p == MAP_FAILED,
                                "mapped_file_t: mmap failed on " + path);
      data_ = static_cast<const char*>(p);

      // The parser is a strict left-to-right scan, so signal the kernel's
      // readahead heuristics that we'll never look back. This typically
      // doubles effective bandwidth on cold-cache reads of multi-GiB files.
      ::madvise(const_cast<char*>(data_), size_, MADV_SEQUENTIAL);
    }
#endif
  }

  ~mapped_file_t() noexcept { release(); }

  mapped_file_t(const mapped_file_t&) = delete;
  mapped_file_t& operator=(const mapped_file_t&) = delete;

  mapped_file_t(mapped_file_t&& rhs) noexcept { steal(rhs); }
  mapped_file_t& operator=(mapped_file_t&& rhs) noexcept {
    if (this != &rhs) {
      release();
      steal(rhs);
    }
    return *this;
  }

  const char* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }
  const char* end() const noexcept { return data_ + size_; }

 private:
  void steal(mapped_file_t& rhs) noexcept {
    data_ = rhs.data_;
    size_ = rhs.size_;
    rhs.data_ = nullptr;
    rhs.size_ = 0;
#if defined(_WIN32)
    file_ = rhs.file_;
    mapping_ = rhs.mapping_;
    rhs.file_ = INVALID_HANDLE_VALUE;
    rhs.mapping_ = nullptr;
#else
    fd_ = rhs.fd_;
    rhs.fd_ = -1;
#endif
  }

  void release() noexcept {
#if defined(_WIN32)
    if (data_)
      UnmapViewOfFile(data_);
    if (mapping_)
      CloseHandle(mapping_);
    if (file_ != INVALID_HANDLE_VALUE)
      CloseHandle(file_);
    data_ = nullptr;
    size_ = 0;
    mapping_ = nullptr;
    file_ = INVALID_HANDLE_VALUE;
#else
    if (data_)
      ::munmap(const_cast<char*>(data_), size_);
    if (fd_ >= 0)
      ::close(fd_);
    data_ = nullptr;
    size_ = 0;
    fd_ = -1;
#endif
  }

  const char* data_ = nullptr;
  std::size_t size_ = 0;
#if defined(_WIN32)
  HANDLE file_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
#else
  int fd_ = -1;
#endif
};

}  // namespace detail
}  // namespace loops
