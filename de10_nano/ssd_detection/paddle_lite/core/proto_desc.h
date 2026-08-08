#pragma once

namespace paddle {
namespace lite {

// The Index of first Block in Program. also called root block.
constexpr int kRootBlockIdx = 0;
// The Parent Index of root Block, this block does not exist.
constexpr int kNoneBlockIdx = -1;

}  // namespace lite
}  // namespace paddle
