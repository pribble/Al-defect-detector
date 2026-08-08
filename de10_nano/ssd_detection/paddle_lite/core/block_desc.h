#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "core/traits.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {

class BlockDescReadAPI {
 public:
  virtual int32_t Idx() const = 0;
  virtual int32_t ParentIdx() const = 0;
  virtual size_t VarsSize() const = 0;
  virtual size_t OpsSize() const = 0;
  virtual int32_t ForwardBlockIdx() const = 0;

  template <typename T>
  T* GetVar(int32_t idx);

  template <typename T>
  T const* GetVar(int32_t idx) const;

  template <typename T>
  T* GetOp(int32_t idx);

  template <typename T>
  T const* GetOp(int32_t idx) const;

  virtual ~BlockDescReadAPI() = default;
};

class BlockDescWriteAPI {
 public:
  virtual void SetIdx(int32_t idx) { LITE_MODEL_INTERFACE_NOT_IMPLEMENTED; }
  virtual void SetParentIdx(int32_t idx) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }
  virtual void ClearVars() { LITE_MODEL_INTERFACE_NOT_IMPLEMENTED; }
  virtual void ClearOps() { LITE_MODEL_INTERFACE_NOT_IMPLEMENTED; }
  virtual void SetForwardBlockIdx(int32_t idx) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }

  template <typename T>
  T* AddVar() {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
    return nullptr;
  }

  template <typename T>
  T* AddOp() {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
    return nullptr;
  }

  virtual ~BlockDescWriteAPI() = default;
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class BlockDescAPI : public BlockDescReadAPI, public BlockDescWriteAPI {
 public:
  virtual ~BlockDescAPI() = default;
};

}  // namespace lite
}  // namespace paddle
