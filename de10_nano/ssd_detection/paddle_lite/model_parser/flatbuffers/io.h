#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>
#include "core/scope.h"
#include "core/variable.h"
#include "model_parser/flatbuffers/param_desc.h"
#include "model_parser/flatbuffers/program_desc.h"

namespace paddle {
namespace lite {
namespace fbs {

void FillParam(const std::string& name,
               const lite::Tensor& tensor,
               ParamDescWriteAPI* prog);

void FillTensor(lite::Tensor* tensor, const ParamDescReadAPI& param);

#ifdef LITE_WITH_FLATBUFFERS_DESC
class ParamSerializer {
 public:
  explicit ParamSerializer(model_parser::ByteWriter* writer,
                           uint16_t version = 0)
      : writer_(writer), version_{version}, buf_(new model_parser::Buffer) {
    CHECK(writer_)
        << "A valid writer should be passed in the ctor of param serializer.";
    WriteHeader();
  }
  void ForwardWrite(const lite::Scope& scope,
                    const std::set<std::string>& param_names);

 private:
  void WriteHeader();
  model_parser::ByteWriter* writer_{nullptr};
  uint16_t version_{0};
  std::unique_ptr<model_parser::Buffer> buf_;
};
#endif

class ParamDeserializer {
 public:
  explicit ParamDeserializer(model_parser::ByteReader* reader)
      : reader_(reader), buf_(new model_parser::Buffer) {
    CHECK(reader_)
        << "A valid reader should be passed in the ctor of param deserializer.";
    ReadHeader();
  }
  void ForwardRead(lite::Scope* scope);

 private:
  void ReadBytesToBuffer(size_t size) {
    buf_->ResetLazy(size);
    reader_->Read(buf_->data(), size);
  }
  void ReadHeader();
  model_parser::ByteReader* reader_{nullptr};
  std::unique_ptr<model_parser::Buffer> buf_;
};

namespace deprecated {
void SetScopeWithCombinedParams(lite::Scope* scope,
                                const CombinedParamsDescReadAPI& params);
void SetCombinedParamsWithScope(const lite::Scope& scope,
                                const std::set<std::string>& param_names,
                                CombinedParamsDescWriteAPI* params);
}  // namespace deprecated

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
