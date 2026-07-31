#pragma once

#include <map>
#include <string>
#include "core/model/base/traits.h"
#include "lite/utils/log/cp_logging.h"
namespace paddle {
namespace lite {

class ProgramDescReadAPI {
 public:
  virtual size_t BlocksSize() const = 0;
  virtual bool HasVersion() const = 0;
  virtual int64_t Version() const = 0;

  virtual bool HasOpVersionMap() const = 0;

  template <typename T>
  T* GetOpVersionMap();

  template <typename T>
  T* GetBlock(int32_t idx);

  template <typename T>
  T const* GetBlock(int32_t idx) const;

  virtual ~ProgramDescReadAPI() = default;
};

class ProgramDescWriteAPI {
 public:
  virtual void ClearBlocks() { LITE_MODEL_INTERFACE_NOT_IMPLEMENTED; }

  virtual void SetVersion(int64_t version) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }

  void SetOpVersionMap(std::map<std::string, int32_t> op_version_map) {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
  }

  template <typename T>
  T* AddBlock() {
    LITE_MODEL_INTERFACE_NOT_IMPLEMENTED;
    return nullptr;
  }

  virtual ~ProgramDescWriteAPI() = default;
};

// The reading and writing of the model are one-time and separate.
// This interface is a combination of reading and writing interfaces,
// which is used to support legacy interfaces.

class ProgramDescAPI : public ProgramDescReadAPI, public ProgramDescWriteAPI {
 public:
  virtual ~ProgramDescAPI() = default;
};

}  // namespace lite
}  // namespace paddle
