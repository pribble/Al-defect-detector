#pragma once

#include <map>
#include "intelfpga.h"  // NOLINT
#include "core/target_wrapper.h"

namespace paddle {
namespace lite {

template <>
class TargetWrapper<TARGET(kIntelFPGA)> {
 public:
  using stream_t = int;
  using event_t = int;

  static size_t num_devices() { return 0; }
  static size_t maximum_stream() { return 0; }

  static void CreateStream(stream_t* stream) {}
  static void DestroyStream(const stream_t& stream) {}

  static void CreateEvent(event_t* event) {}
  static void DestroyEvent(const event_t& event) {}

  static void RecordEvent(const event_t& event) {}
  static void SyncEvent(const event_t& event) {}

  static void StreamSync(const stream_t& stream) {}

  static void* Malloc(size_t size);
  static void Free(void* ptr);

  static void MemcpySync(void* dst,
                         const void* src,
                         size_t size,
                         IoDirection dir);
  static void MemcpyAsync(void* dst,
                          const void* src,
                          size_t size,
                          IoDirection dir,
                          const stream_t& stream) {
    MemcpySync(dst, src, size, dir);
  }
};

}  // namespace lite
}  // namespace paddle
