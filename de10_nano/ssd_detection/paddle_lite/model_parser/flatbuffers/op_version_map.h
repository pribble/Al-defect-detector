#pragma once

#include <map>
#include <string>
#include "core/apis.h"
#include "model_parser/flatbuffers/framework_generated.h"

namespace paddle {
namespace lite {
namespace fbs {

/*
 * The general::OpVersionMap is the internal representation for Ops version.
 * All the internal
 * imprementation should use it, not the pb::OpVersionMap.
 */
class OpVersionMap : public OpVersionMapAPI {
 public:
  OpVersionMap() = default;

  explicit OpVersionMap(proto::OpVersionMap* op_version_map) {
    // op_version_map is not implemented on flatbuffer as
    // it's not useful in inference period.
  }
  std::map<std::string, int32_t> GetOpVersionMap() const override {
    return op_version_map_;
  }
  int32_t GetOpVersionByName(const std::string& name) const override {
    return op_version_map_.at(name);
  }

  void SetOpVersionMap(
      const std::map<std::string, int32_t>& op_version_map) override {
    op_version_map_ = op_version_map;
  }

  void AddOpVersion(const std::string& op_name, int32_t op_version) override {
    op_version_map_[op_name] = op_version;
  }

 private:
  std::map<std::string, int32_t> op_version_map_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
