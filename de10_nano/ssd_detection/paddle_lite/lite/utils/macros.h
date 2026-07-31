#pragma once

// UNUSED — suppresses "unused variable" warnings in GCC/Clang
#if defined(_WIN32)
#define UNUSED
#define __builtin_expect(EXP, C) (EXP)
#else
#define UNUSED __attribute__((unused))
#endif
