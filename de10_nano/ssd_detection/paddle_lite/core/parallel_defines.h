#pragma once

#include <tuple>
#include <utility>
#include "core/thread_pool.h"

#include <omp.h>

#define LITE_PARALLEL_BEGIN(index, tid, work_size)                     \
  _Pragma("omp parallel for") for (int index = 0; index < (work_size); \
                                   ++index) {
#define LITE_PARALLEL_END() }

#define LITE_PARALLEL_COMMON_BEGIN(index, tid, end, start, step)       \
  _Pragma("omp parallel for") for (int index = (start); index < (end); \
                                   index += (step)) {
#define LITE_PARALLEL_COMMON_END() }

