#pragma once

#include <cstdio>
#include <cstdlib>
#include <ctime>

namespace iris {
namespace rdma {

// Log levels
enum class log_level {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  NONE = 4
};

// Global log level (can be set via environment variable)
inline log_level get_log_level() {
  static log_level level = []() {
    const char* env = std::getenv("IRIS_LOG_LEVEL");
    if (!env) return log_level::INFO;
    
    if (strcmp(env, "DEBUG") == 0) return log_level::DEBUG;
    if (strcmp(env, "INFO") == 0) return log_level::INFO;
    if (strcmp(env, "WARN") == 0) return log_level::WARN;
    if (strcmp(env, "ERROR") == 0) return log_level::ERROR;
    if (strcmp(env, "NONE") == 0) return log_level::NONE;
    
    return log_level::INFO;
  }();
  return level;
}

// Check if debug data printing is enabled (separate from log level)
inline bool is_debug_data_enabled() {
  static bool enabled = (std::getenv("IRIS_DEBUG_DATA") != nullptr);
  return enabled;
}

// Internal logging function
inline void log_message(log_level level, const char* level_str, const char* fmt, ...) {
  if (level < get_log_level()) return;
  
  // Get timestamp
  time_t now = time(nullptr);
  struct tm* tm_info = localtime(&now);
  char time_buf[64];
  strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);
  
  // Print level and timestamp
  fprintf(stderr, "[%s] [%s] ", time_buf, level_str);
  
  // Print message
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  
  fprintf(stderr, "\n");
  fflush(stderr);
}

// Internal logging function with rank
inline void log_message_rank(int rank, log_level level, const char* level_str, const char* fmt, ...) {
  if (level < get_log_level()) return;
  
  // Get timestamp
  time_t now = time(nullptr);
  struct tm* tm_info = localtime(&now);
  char time_buf[64];
  strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);
  
  // Print level, timestamp, and rank
  fprintf(stderr, "[%s] [%s] [RANK %d] ", time_buf, level_str, rank);
  
  // Print message
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  
  fprintf(stderr, "\n");
  fflush(stderr);
}

}  // namespace rdma
}  // namespace iris

// Logging macros - easy to replace with real logging library later
#define LOG_DEBUG(fmt, ...) \
  iris::rdma::log_message(iris::rdma::log_level::DEBUG, "DEBUG", fmt, ##__VA_ARGS__)

#define LOG_INFO(fmt, ...) \
  iris::rdma::log_message(iris::rdma::log_level::INFO, "INFO", fmt, ##__VA_ARGS__)

#define LOG_WARN(fmt, ...) \
  iris::rdma::log_message(iris::rdma::log_level::WARN, "WARN", fmt, ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) \
  iris::rdma::log_message(iris::rdma::log_level::ERROR, "ERROR", fmt, ##__VA_ARGS__)

// Rank-aware logging macros
#define LOG_DEBUG_RANK(rank, fmt, ...) \
  iris::rdma::log_message_rank(rank, iris::rdma::log_level::DEBUG, "DEBUG", fmt, ##__VA_ARGS__)

#define LOG_INFO_RANK(rank, fmt, ...) \
  iris::rdma::log_message_rank(rank, iris::rdma::log_level::INFO, "INFO", fmt, ##__VA_ARGS__)

#define LOG_WARN_RANK(rank, fmt, ...) \
  iris::rdma::log_message_rank(rank, iris::rdma::log_level::WARN, "WARN", fmt, ##__VA_ARGS__)

#define LOG_ERROR_RANK(rank, fmt, ...) \
  iris::rdma::log_message_rank(rank, iris::rdma::log_level::ERROR, "ERROR", fmt, ##__VA_ARGS__)

// For data debugging (separate from regular logging)
#define LOG_DATA_DEBUG(fmt, ...) \
  do { \
    if (iris::rdma::is_debug_data_enabled()) { \
      fprintf(stderr, "[DEBUG-DATA] " fmt "\n", ##__VA_ARGS__); \
      fflush(stderr); \
    } \
  } while (0)

