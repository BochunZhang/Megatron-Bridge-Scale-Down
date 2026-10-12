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

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gb {

[[noreturn]] inline void fail(const std::string& message) { throw std::runtime_error(message); }

inline void check_cuda(cudaError_t err, const char* expr, const char* file, int line) {
  if (err == cudaSuccess) {
    return;
  }
  std::ostringstream os;
  os << expr << " failed at " << file << ":" << line << ": " << cudaGetErrorName(err) << " ("
     << cudaGetErrorString(err) << ")";
  fail(os.str());
}

#define GB_CHECK_CUDA(expr) ::gb::check_cuda((expr), #expr, __FILE__, __LINE__)

inline uint64_t now_ns() {
  timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

inline void cpu_relax() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__)
  __builtin_ia32_pause();
#endif
}

// Written by benchmark loops so that the compiler cannot drop their loads.
inline volatile uint64_t g_sink = 0;

// Value stored in 64-bit word `index` of a pattern-filled buffer; used to verify copies.
__host__ __device__ inline uint64_t pattern_word(uint64_t index, uint64_t seed) {
  return (index * 0x9E3779B97F4A7C15ull) ^ seed;
}

// ---------------------------------------------------------------- strings

inline std::string trim(const std::string& text) {
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
  return text.substr(begin, end - begin);
}

inline std::vector<std::string> split(const std::string& text, char sep) {
  std::vector<std::string> items;
  std::string item;
  std::istringstream is(text);
  while (std::getline(is, item, sep)) {
    item = trim(item);
    if (!item.empty()) items.push_back(item);
  }
  return items;
}

inline bool starts_with(const std::string& text, const std::string& prefix) {
  return text.compare(0, prefix.size(), prefix) == 0;
}

inline std::string lower(std::string text) {
  for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return text;
}

inline long long parse_int(const std::string& text, const std::string& what) {
  char* end = nullptr;
  errno = 0;
  long long value = std::strtoll(text.c_str(), &end, 10);
  if (text.empty() || errno != 0 || *end != '\0') fail("invalid integer for " + what + ": '" + text + "'");
  return value;
}

// Accepts plain bytes or binary units: 64K, 64KiB, 256M, 4G, 4GiB, 1T.
inline size_t parse_size(const std::string& text) {
  size_t pos = 0;
  double value = 0;
  try {
    value = std::stod(text, &pos);
  } catch (const std::exception&) {
    fail("invalid size '" + text + "'");
  }
  const std::string unit = lower(trim(text.substr(pos)));
  double multiplier = 1;
  if (unit.empty() || unit == "b") {
    multiplier = 1;
  } else if (unit == "k" || unit == "kb" || unit == "kib") {
    multiplier = 1024.0;
  } else if (unit == "m" || unit == "mb" || unit == "mib") {
    multiplier = 1024.0 * 1024.0;
  } else if (unit == "g" || unit == "gb" || unit == "gib") {
    multiplier = 1024.0 * 1024.0 * 1024.0;
  } else if (unit == "t" || unit == "tb" || unit == "tib") {
    multiplier = 1024.0 * 1024.0 * 1024.0 * 1024.0;
  } else {
    fail("invalid size unit in '" + text + "'");
  }
  if (value <= 0) fail("size must be positive: '" + text + "'");
  return static_cast<size_t>(value * multiplier);
}

inline std::string format_bytes(size_t bytes) {
  static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  double value = static_cast<double>(bytes);
  int unit = 0;
  while (value >= 1024.0 && unit < 4) {
    value /= 1024.0;
    ++unit;
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4g %s", value, units[unit]);
  return buf;
}

// {0,1,2,3,8} -> "0-3,8"
inline std::string compress_list(std::vector<int> values) {
  std::sort(values.begin(), values.end());
  std::ostringstream os;
  for (size_t i = 0; i < values.size();) {
    size_t j = i;
    while (j + 1 < values.size() && values[j + 1] == values[j] + 1) ++j;
    if (i != 0) os << ",";
    os << values[i];
    if (j > i) os << "-" << values[j];
    i = j + 1;
  }
  return os.str();
}

inline std::string timestamp_iso() {
  char buf[64];
  std::time_t t = std::time(nullptr);
  std::tm tm_local;
  localtime_r(&t, &tm_local);
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S%z", &tm_local);
  return buf;
}

// ---------------------------------------------------------------- statistics

struct Stats {
  double min = NAN;
  double max = NAN;
  double mean = NAN;
  double median = NAN;
  double stddev = NAN;
};

inline Stats summarize(std::vector<double> values) {
  Stats s;
  if (values.empty()) return s;
  std::sort(values.begin(), values.end());
  s.min = values.front();
  s.max = values.back();
  const size_t n = values.size();
  s.median = n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
  double sum = 0;
  for (double v : values) sum += v;
  s.mean = sum / n;
  double var = 0;
  for (double v : values) var += (v - s.mean) * (v - s.mean);
  s.stddev = std::sqrt(var / n);
  return s;
}

// ---------------------------------------------------------------- JSON

inline std::string json_quote(const std::string& text) {
  std::string out = "\"";
  for (char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out + "\"";
}

inline std::string json_number(double value) {
  if (!std::isfinite(value)) return "null";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.6g", value);
  return buf;
}

inline std::string json_array(const std::vector<std::string>& raw_items) {
  std::string out = "[";
  for (size_t i = 0; i < raw_items.size(); ++i) {
    if (i) out += ",";
    out += raw_items[i];
  }
  return out + "]";
}

inline std::string json_numbers(const std::vector<double>& values) {
  std::vector<std::string> items;
  for (double v : values) items.push_back(json_number(v));
  return json_array(items);
}

inline std::string json_ints(const std::vector<int>& values) {
  std::vector<std::string> items;
  for (int v : values) items.push_back(std::to_string(v));
  return json_array(items);
}

// Insertion-ordered JSON object builder.
class JsonObject {
 public:
  JsonObject& str(const std::string& key, const std::string& value) { return raw(key, json_quote(value)); }
  JsonObject& num(const std::string& key, double value) { return raw(key, json_number(value)); }
  JsonObject& integer(const std::string& key, long long value) { return raw(key, std::to_string(value)); }
  JsonObject& boolean(const std::string& key, bool value) { return raw(key, value ? "true" : "false"); }
  JsonObject& raw(const std::string& key, const std::string& json) {
    fields_.emplace_back(key, json);
    return *this;
  }
  JsonObject& merge(const JsonObject& other) {
    fields_.insert(fields_.end(), other.fields_.begin(), other.fields_.end());
    return *this;
  }
  std::string dump() const {
    std::string out = "{";
    for (size_t i = 0; i < fields_.size(); ++i) {
      if (i) out += ",";
      out += json_quote(fields_[i].first) + ":" + fields_[i].second;
    }
    return out + "}";
  }

 private:
  std::vector<std::pair<std::string, std::string>> fields_;
};

inline std::string json_stats(const Stats& s) {
  return JsonObject()
      .num("min", s.min)
      .num("median", s.median)
      .num("mean", s.mean)
      .num("max", s.max)
      .num("stddev", s.stddev)
      .dump();
}

// ---------------------------------------------------------------- CLI

// `--key value` / `--key=value` options. Every option takes a value; unknown keys are rejected by finish().
class Args {
 public:
  Args(int argc, char** argv, int first) {
    for (int i = first; i < argc; ++i) {
      std::string token = argv[i];
      if (!starts_with(token, "--")) fail("unexpected argument '" + token + "'");
      token = token.substr(2);
      const size_t eq = token.find('=');
      if (eq != std::string::npos) {
        values_[token.substr(0, eq)] = token.substr(eq + 1);
      } else {
        if (i + 1 >= argc) fail("option --" + token + " needs a value");
        values_[token] = argv[++i];
      }
    }
  }

  bool has(const std::string& key) const {
    used_.insert(key);
    return values_.count(key) != 0;
  }

  std::string get(const std::string& key, const std::string& fallback) const {
    used_.insert(key);
    auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
  }

  int get_int(const std::string& key, int fallback) const {
    return has(key) ? static_cast<int>(parse_int(get(key, ""), "--" + key)) : fallback;
  }

  void finish() const {
    for (const auto& kv : values_) {
      if (!used_.count(kv.first)) fail("unknown option --" + kv.first + " for this command (see --help)");
    }
  }

 private:
  std::map<std::string, std::string> values_;
  mutable std::set<std::string> used_;
};

}  // namespace gb
