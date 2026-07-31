#pragma once

#include <omp.h>
#include "core/tensor.h"

namespace paddle {
namespace lite {

typedef enum {
  kAPPLE = 0, kX1 = 1, kX2 = 2,
  kA35 = 35, kA53 = 53, kA55 = 55, kA57 = 57,
  kA510 = 60, kA72 = 72, kA73 = 73, kA75 = 75,
  kA76 = 76, kA77 = 77, kA78 = 78,
  kGold = 79, kGold_Prime = 80, kSilver = 81, kA710 = 82,
  kARMArch_UNKOWN = -1
} ARMArch;

#define CORTEX_A9_CORES 2

// ── DeviceInfo ────────────────────────────────────────────
// Cyclone-V-hardcoded singleton.  ARM math backends access
// thread count, cache sizes, workspace and feature flags via
// this class (or through the ARMContext / KernelContext
// aliases below).

class DeviceInfo {
 public:
  static DeviceInfo& Global() {
    static DeviceInfo x;
    return x;
  }

  int threads() const { return CORTEX_A9_CORES; }
  ARMArch arch() const { return kA53; }
  int llc_size() const { return 512 * 1024; }

  bool has_dot() const { return false; }
  bool has_fp16() const { return false; }
  bool has_a53_valid() const { return true; }
  bool has_sve2() const { return false; }
  bool has_sve2_i8mm() const { return false; }
  bool has_sve2_f32mm() const { return false; }

  template <typename T>
  T* workspace_data() {
    workspace_.Resize({llc_size()});
    return workspace_.mutable_data<T>();
  }
  bool ExtendWorkspace(size_t size) {
    workspace_.Resize({static_cast<int64_t>(size + llc_size())});
    return workspace_.mutable_data<int8_t>() != nullptr;
  }

 private:
  DeviceInfo() { omp_set_num_threads(CORTEX_A9_CORES); }
  TensorLite workspace_;
};

// ── KernelContext ─────────────────────────────────────────
// Shell owned by each KernelBase via unique_ptr.  As<T>()
// returns DeviceInfo::Global() directly — no per-kernel state.

class KernelContext {
 public:
  template <typename ContextT>
  DeviceInfo& As() { return DeviceInfo::Global(); }
};

using HostContext = DeviceInfo;
using ARMContext = DeviceInfo;
using IntelFPGAContext = DeviceInfo;

template <TargetType Type>
using Context = DeviceInfo;

class ContextScheduler {
 public:
  static ContextScheduler& Global() {
    static auto* x = new ContextScheduler;
    return *x;
  }
  std::unique_ptr<KernelContext> NewContext(TargetType target) {
    return std::unique_ptr<KernelContext>(new KernelContext);
  }
 private:
  ContextScheduler() = default;
};

}  // namespace lite
}  // namespace paddle
