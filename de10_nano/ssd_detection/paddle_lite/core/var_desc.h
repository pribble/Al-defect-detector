#pragma once

#include <string>
#include <vector>
#include "core/traits.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

class VarDescReadAPI {
 public:
  virtual std::string Name() const = 0;
  virtual VarDataType GetType() const = 0;
  virtual bool Persistable() const = 0;
  virtual std::vector<int64_t> GetShape() const = 0;
  virtual ~VarDescReadAPI() = default;
};

class VarDescWriteAPI {
 public:
  virtual void SetName(std::string name) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual void SetType(VarDataType type) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual void SetPersistable(bool persistable) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual void SetShape(const std::vector<int64_t>& dims) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual ~VarDescWriteAPI() = default;
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class VarDescAPI : public VarDescReadAPI, public VarDescWriteAPI {
 public:
  using VarDataType = lite::VarDataType;
  using Type = lite::VarDataType;
  virtual ~VarDescAPI() = default;
};

inline bool IsParamVarDesc(const VarDescReadAPI& var) {
  return var.GetType() == VarDataType::LOD_TENSOR && var.Persistable();
}

}  // namespace lite
}  // namespace paddle
