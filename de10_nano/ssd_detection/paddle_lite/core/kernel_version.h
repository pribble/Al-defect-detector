#pragma once
#include <map>
#include <string>
#include "lite/utils/log/cp_logging.h"
namespace paddle {
namespace lite {

///////////////////////////////////////////////////////////
// Name: KernelVersion
// Description: Version of Paddle-Lite kernel
//              (a list of OpVersion)
///////////////////////////////////////////////////////////

class KernelVersion {
 public:
  // Fill op_versions into kernel_version
  void AddOpVersion(const std::string& name, int32_t op_version) {
    if (!op_versions_.count(name)) {
      op_versions_[name] = op_version;
    } else {
      LOG(FATAL) << "Error: binding kernel to the version of op(" << name
                 << ") more than once is not allowed.";
    }
  }
  // Return the content of kernel_version: list(op_version)
  const std::map<std::string, int32_t>& OpVersions() const {
    return op_versions_;
  }
  // Judge if an op_version has been bound to this kernel.
  bool HasOpVersion(const std::string& op_name) {
    return op_versions_.count(op_name);
  }

  // Get a inner op_version according to op_name.
  int32_t GetOpVersion(const std::string& op_name) {
    if (HasOpVersion(op_name)) {
      return op_versions_[op_name];
    } else {
      LOG(FATAL) << "Error: This kernel has not been bound to Paddle op ("
                 << op_name << ") 's version.";
      return -1;
    }
  }

 private:
  // Paddle OpVersion: Version of Paddle operator
  //              (op_name + version_id)
  std::map<std::string, int32_t> op_versions_;
};

}  // namespace lite
}  // namespace paddle
