#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "core/traits.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

class OpVersionMapReadAPI {
 public:
  virtual std::map<std::string, int32_t> GetOpVersionMap() const = 0;
  virtual int32_t GetOpVersionByName(const std::string& name) const = 0;
  virtual ~OpVersionMapReadAPI() = default;
};

class OpVersionMapWriteAPI {
 public:
  virtual void SetOpVersionMap(
      const std::map<std::string, int32_t>& op_version_map) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual void AddOpVersion(const std::string& op_name, int32_t op_version) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual ~OpVersionMapWriteAPI() = default;
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class OpVersionMapAPI : public OpVersionMapReadAPI,
                        public OpVersionMapWriteAPI {
 public:
  virtual ~OpVersionMapAPI() = default;
};

}  // namespace lite
}  // namespace paddle
