#pragma once

#define UNUSED __attribute__((unused))

#define USE_LITE_OP(op_type__)       \
  extern int touch_op_##op_type__(); \
  int LITE_OP_REGISTER_FAKE(op_type__) UNUSED = touch_op_##op_type__();

#define USE_LITE_KERNEL(op_type__, target__, precision__, layout__, alias__) \
  extern int touch_##op_type__##target__##precision__##layout__##alias__();  \
  int op_type__##target__##precision__##layout__##alias__##__use_lite_kernel \
      UNUSED = touch_##op_type__##target__##precision__##layout__##alias__();

#define LITE_OP_REGISTER_FAKE(op_type__) op_type__##__registry__
