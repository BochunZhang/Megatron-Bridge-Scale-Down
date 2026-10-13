// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <signal.h>

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

#include "common.hpp"

namespace gb {

constexpr size_t kLine = 64;

// ---------------------------------------------------------------- CPU access probe

inline sigjmp_buf g_fault_jump;
inline volatile sig_atomic_t g_fault_probing = 0;

inline void on_probe_fault(int sig) {
  if (g_fault_probing) siglongjmp(g_fault_jump, sig);
  signal(sig, SIG_DFL);
  raise(sig);
}

// Loads and stores back one word from the calling thread. Returns "" on success, otherwise the signal that the
// access raised (e.g. a CPU dereferencing GPU memory that is not mapped for it).
inline std::string cpu_touch(uint64_t* p, uint64_t* loaded) {
  struct sigaction sa = {};
  struct sigaction old_segv = {};
  struct sigaction old_bus = {};
  sa.sa_handler = on_probe_fault;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGSEGV, &sa, &old_segv);
  sigaction(SIGBUS, &sa, &old_bus);
  std::string result;
  const int sig = sigsetjmp(g_fault_jump, 1);
  if (sig == 0) {
    g_fault_probing = 1;
    volatile uint64_t* vp = p;
    const uint64_t v = *vp;
    *vp = v;
    *loaded = v;
  } else {
    result = sig == SIGSEGV ? "SIGSEGV" : "SIGBUS";
  }
  g_fault_probing = 0;
  sigaction(SIGSEGV, &old_segv, nullptr);
  sigaction(SIGBUS, &old_bus, nullptr);
  return result;
}

enum class HostOp { Read, Write, Copy };

inline HostOp parse_op(const std::string& text) {
  if (text == "read") return HostOp::Read;
  if (text == "write") return HostOp::Write;
  if (text == "copy") return HostOp::Copy;
  fail("--op must be read, write or copy, got '" + text + "'");
}

inline const char* op_name(HostOp op) {
  switch (op) {
    case HostOp::Read:
      return "read";
    case HostOp::Write:
      return "write";
    default:
      return "copy";
  }
}

// CPU load/store loops. ldp_stp is the gh_benchmark loop (64 B per iteration with ldp/stp on x registers);
// ldnp_stnp uses non-temporal pairs of 128-bit registers; memcpy uses glibc memcpy/memset.
enum class HostKernel { LdpStp, LdnpStnp, Memcpy };

inline HostKernel parse_kernel(const std::string& text) {
  if (text == "ldp_stp") return HostKernel::LdpStp;
  if (text == "ldnp_stnp") return HostKernel::LdnpStnp;
  if (text == "memcpy") return HostKernel::Memcpy;
  fail("host kernel must be ldp_stp, ldnp_stnp or memcpy, got '" + text + "'");
}

inline const char* kernel_name(HostKernel k) {
  switch (k) {
    case HostKernel::LdpStp:
      return "ldp_stp";
    case HostKernel::LdnpStnp:
      return "ldnp_stnp";
    default:
      return "memcpy";
  }
}

namespace hostk {

// Portable fallbacks, used for the memcpy kernel and on non-AArch64 build hosts.
inline void generic_copy(const uint8_t* src, uint8_t* dst, size_t bytes) { std::memcpy(dst, src, bytes); }

inline void generic_read(const uint8_t* src, size_t bytes) {
  const auto* p = reinterpret_cast<const volatile uint64_t*>(src);
  uint64_t acc = 0;
  for (size_t i = 0; i < bytes / 8; i += 8) acc ^= p[i] ^ p[i + 1] ^ p[i + 2] ^ p[i + 3] ^ p[i + 4] ^ p[i + 5] ^ p[i + 6] ^ p[i + 7];
  g_sink = acc;
}

inline void generic_write(uint8_t* dst, size_t bytes, uint64_t value) {
  auto* p = reinterpret_cast<uint64_t*>(dst);
  for (size_t i = 0; i < bytes / 8; ++i) p[i] = value;
}

inline void copy(HostKernel k, const uint8_t* src, uint8_t* dst, size_t bytes) {
#if defined(__aarch64__)
  if (k == HostKernel::LdpStp) {
    for (size_t off = 0; off < bytes; off += kLine) {
      asm volatile(
          "ldp x0, x1, [%0]\n\t"
          "stp x0, x1, [%1]\n\t"
          "ldp x2, x3, [%0, #16]\n\t"
          "stp x2, x3, [%1, #16]\n\t"
          "ldp x4, x5, [%0, #32]\n\t"
          "stp x4, x5, [%1, #32]\n\t"
          "ldp x6, x7, [%0, #48]\n\t"
          "stp x6, x7, [%1, #48]\n\t"
          :
          : "r"(src + off), "r"(dst + off)
          : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "memory");
    }
    return;
  }
  if (k == HostKernel::LdnpStnp) {
    for (size_t off = 0; off < bytes; off += kLine) {
      asm volatile(
          "ldnp q0, q1, [%0]\n\t"
          "ldnp q2, q3, [%0, #32]\n\t"
          "stnp q0, q1, [%1]\n\t"
          "stnp q2, q3, [%1, #32]\n\t"
          :
          : "r"(src + off), "r"(dst + off)
          : "v0", "v1", "v2", "v3", "memory");
    }
    return;
  }
#else
  (void)k;
#endif
  generic_copy(src, dst, bytes);
}

inline void read(HostKernel k, const uint8_t* src, size_t bytes) {
#if defined(__aarch64__)
  if (k == HostKernel::LdpStp) {
    for (size_t off = 0; off < bytes; off += kLine) {
      asm volatile(
          "ldp x0, x1, [%0]\n\t"
          "ldp x2, x3, [%0, #16]\n\t"
          "ldp x4, x5, [%0, #32]\n\t"
          "ldp x6, x7, [%0, #48]\n\t"
          :
          : "r"(src + off)
          : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "memory");
    }
    return;
  }
  if (k == HostKernel::LdnpStnp) {
    for (size_t off = 0; off < bytes; off += kLine) {
      asm volatile(
          "ldnp q0, q1, [%0]\n\t"
          "ldnp q2, q3, [%0, #32]\n\t"
          :
          : "r"(src + off)
          : "v0", "v1", "v2", "v3", "memory");
    }
    return;
  }
#else
  (void)k;
#endif
  generic_read(src, bytes);
}

// Non-zero value: Grace treats all-zero stores specially (see gh_benchmark's memset note).
constexpr uint64_t kWriteValue = 0xA5A5A5A5A5A5A5A5ull;

inline void write(HostKernel k, uint8_t* dst, size_t bytes) {
#if defined(__aarch64__)
  if (k == HostKernel::LdpStp) {
    const uint64_t v = kWriteValue;
    for (size_t off = 0; off < bytes; off += kLine) {
      asm volatile(
          "stp %1, %1, [%0]\n\t"
          "stp %1, %1, [%0, #16]\n\t"
          "stp %1, %1, [%0, #32]\n\t"
          "stp %1, %1, [%0, #48]\n\t"
          :
          : "r"(dst + off), "r"(v)
          : "memory");
    }
    return;
  }
  if (k == HostKernel::LdnpStnp) {
    const uint64_t v = kWriteValue;
    for (size_t off = 0; off < bytes; off += kLine) {
      asm volatile(
          "dup v0.2d, %1\n\t"
          "stnp q0, q0, [%0]\n\t"
          "stnp q0, q0, [%0, #32]\n\t"
          :
          : "r"(dst + off), "r"(v)
          : "v0", "memory");
    }
    return;
  }
#endif
  if (k == HostKernel::Memcpy) {
    std::memset(dst, 0xA5, bytes);
  } else {
    generic_write(dst, bytes, kWriteValue);
  }
}

inline void fill_pattern(uint8_t* dst, size_t bytes, uint64_t first_word, uint64_t seed) {
  auto* p = reinterpret_cast<uint64_t*>(dst);
  for (size_t i = 0; i < bytes / 8; ++i) p[i] = pattern_word(first_word + i, seed);
}

inline size_t count_mismatch(const uint8_t* src, size_t bytes, uint64_t first_word, uint64_t seed) {
  const auto* p = reinterpret_cast<const uint64_t*>(src);
  size_t bad = 0;
  for (size_t i = 0; i < bytes / 8; ++i) bad += p[i] != pattern_word(first_word + i, seed);
  return bad;
}

}  // namespace hostk

// Persistent worker threads, each pinned to one CPU. run(fn, n) executes fn(tid) on workers [0, n).
class HostThreadPool {
 public:
  explicit HostThreadPool(std::vector<int> cpus) : cpus_(std::move(cpus)), pinned_(cpus_.size(), 0) {
    if (cpus_.empty()) fail("no CPUs available for the host thread pool (check the process CPU affinity)");
    for (size_t i = 0; i < cpus_.size(); ++i) workers_.emplace_back([this, i] { worker(i); });
    run([](size_t) {}, cpus_.size());  // wait until every worker is up and pinned
  }

  ~HostThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
      ++generation_;
    }
    cv_.notify_all();
    for (std::thread& t : workers_) t.join();
  }

  HostThreadPool(const HostThreadPool&) = delete;
  HostThreadPool& operator=(const HostThreadPool&) = delete;

  size_t size() const { return cpus_.size(); }
  const std::vector<int>& cpus() const { return cpus_; }

  size_t pin_failures() const {
    size_t n = 0;
    for (char ok : pinned_) n += !ok;
    return n;
  }

  void run(const std::function<void(size_t)>& fn, size_t n_active) {
    if (n_active == 0 || n_active > cpus_.size()) fail("invalid number of active pool threads");
    std::unique_lock<std::mutex> lock(mu_);
    task_ = &fn;
    active_ = n_active;
    remaining_ = n_active;
    ++generation_;
    cv_.notify_all();
    done_cv_.wait(lock, [this] { return remaining_ == 0; });
    task_ = nullptr;
  }

 private:
  void worker(size_t tid) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus_[tid], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    pinned_[tid] = sched_getcpu() == cpus_[tid];
    uint64_t seen = 0;
    for (;;) {
      const std::function<void(size_t)>* task = nullptr;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return generation_ != seen; });
        seen = generation_;
        if (stop_) return;
        if (tid >= active_) continue;
        task = task_;
      }
      (*task)(tid);
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (--remaining_ == 0) done_cv_.notify_one();
      }
    }
  }

  std::vector<int> cpus_;
  std::vector<char> pinned_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  const std::function<void(size_t)>* task_ = nullptr;
  size_t active_ = 0;
  size_t remaining_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

struct Chunk {
  size_t offset;
  size_t bytes;
};

// Splits `bytes` (rounded down to whole 64 B lines) into `parts` contiguous chunks.
inline std::vector<Chunk> split_lines(size_t bytes, size_t parts) {
  const size_t lines = bytes / kLine;
  std::vector<Chunk> chunks;
  size_t offset = 0;
  for (size_t i = 0; i < parts; ++i) {
    const size_t n = lines / parts + (i < lines % parts ? 1 : 0);
    chunks.push_back({offset * kLine, n * kLine});
    offset += n;
  }
  return chunks;
}

inline void parallel_fill(HostThreadPool& pool, uint8_t* buf, size_t bytes, uint64_t seed) {
  const auto chunks = split_lines(bytes, pool.size());
  pool.run([&](size_t t) { hostk::fill_pattern(buf + chunks[t].offset, chunks[t].bytes, chunks[t].offset / 8, seed); },
           pool.size());
}

inline size_t parallel_count_mismatch(HostThreadPool& pool, const uint8_t* buf, size_t bytes, uint64_t seed) {
  const auto chunks = split_lines(bytes, pool.size());
  std::vector<size_t> bad(pool.size(), 0);
  pool.run(
      [&](size_t t) { bad[t] = hostk::count_mismatch(buf + chunks[t].offset, chunks[t].bytes, chunks[t].offset / 8, seed); },
      pool.size());
  size_t total = 0;
  for (size_t b : bad) total += b;
  return total;
}

struct HostTiming {
  double ms = 0;
  double max_late_us = 0;  // how long after the common start time the slowest thread began
};

// All threads spin until a common start time, run their chunk, and the elapsed time is measured from that start
// to the last thread's end (same scheme as gh_benchmark's run_test/time_test).
inline HostTiming timed_host_op(HostThreadPool& pool, size_t n_threads, HostOp op, HostKernel kernel,
                                const uint8_t* src, uint8_t* dst, size_t bytes) {
  const auto chunks = split_lines(bytes, n_threads);
  std::vector<uint64_t> begin(n_threads, 0);
  std::vector<uint64_t> end(n_threads, 0);
  const uint64_t start = now_ns() + 20000000;  // 20 ms ahead so that every worker is spinning before the start
  pool.run(
      [&](size_t t) {
        const Chunk& c = chunks[t];
        while (now_ns() < start) cpu_relax();
        begin[t] = now_ns();
        switch (op) {
          case HostOp::Read:
            hostk::read(kernel, src + c.offset, c.bytes);
            break;
          case HostOp::Write:
            hostk::write(kernel, dst + c.offset, c.bytes);
            break;
          case HostOp::Copy:
            hostk::copy(kernel, src + c.offset, dst + c.offset, c.bytes);
            break;
        }
        end[t] = now_ns();
      },
      n_threads);
  HostTiming timing;
  uint64_t last = start;
  for (size_t t = 0; t < n_threads; ++t) {
    last = std::max(last, end[t]);
    timing.max_late_us = std::max(timing.max_late_us, (static_cast<double>(begin[t]) - start) / 1e3);
  }
  timing.ms = static_cast<double>(last - start) / 1e6;
  return timing;
}

}  // namespace gb
