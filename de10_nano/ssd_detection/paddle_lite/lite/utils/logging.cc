#include "lite/utils/logging.h"
#include <iomanip>

#if defined(LITE_WITH_ARM) || defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || \
    defined(LITE_WITH_PYTHON) || defined(LITE_WITH_XPU)
#ifdef LITE_WITH_LOG

namespace paddle {
namespace lite {

void gen_log(STL::ostream& log_stream_,
             const char* file,
             const char* func,
             int lineno,
             const char* level,
             const int kMaxLen) {
  const int len = strlen(file);

  struct tm tm_time;  // Time of creation of LogMessage
  time_t timestamp = time(NULL);
#if defined(_WIN32)
  localtime_s(&tm_time, &timestamp);
#else
  localtime_r(&timestamp, &tm_time);
#endif
  struct timeval tv;
  gettimeofday(&tv, NULL);

  // print date / time
  log_stream_ << '[' << level << ' ' << STL::setw(2) << 1 + tm_time.tm_mon
              << '/' << STL::setw(2) << tm_time.tm_mday << ' ' << STL::setw(2)
              << tm_time.tm_hour << ':' << STL::setw(2) << tm_time.tm_min << ':'
              << STL::setw(2) << tm_time.tm_sec << '.' << STL::setw(3)
              << tv.tv_usec / 1000 << " ";

  if (len > kMaxLen) {
    log_stream_ << "..." << file + len - kMaxLen << ":" << lineno << " " << func
                << "] ";
  } else {
    log_stream_ << file << " " << func << ":" << lineno << "] ";
  }
}

}  // namespace lite
}  // namespace paddle

#endif  // LITE_WITH_LOG
#endif  // LITE_WITH_ARM
