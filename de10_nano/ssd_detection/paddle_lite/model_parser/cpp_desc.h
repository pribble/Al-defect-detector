#pragma once

#include "model_parser/flatbuffers/block_desc.h"
#include "model_parser/flatbuffers/op_desc.h"
#include "model_parser/flatbuffers/program_desc.h"
#include "model_parser/flatbuffers/var_desc.h"

namespace paddle {
namespace lite {
namespace cpp {

using ProgramDesc = fbs::ProgramDescView;
using BlockDesc = fbs::BlockDescView;
using OpDesc = fbs::OpDescView;
using VarDesc = fbs::VarDescView;
using OpDescWrite = fbs::OpDesc;

}  // namespace cpp
}  // namespace lite
}  // namespace paddle
