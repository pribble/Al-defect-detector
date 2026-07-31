# x86 SIMD detection — unused for ARM cross-compile (WITH_AVX=OFF).
# Retained as a stub for cmake module completeness.
if(WITH_AVX)
  include(CheckCXXSourceRuns)
  if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(AVX_FLAG "-mavx")
  endif()
  CHECK_CXX_SOURCE_RUNS("
  #include <immintrin.h>
  int main() {
    __m256 a = _mm256_set_ps (-1.0f, 2.0f, -3.0f, 4.0f, -1.0f, 2.0f, -3.0f, 4.0f);
    __m256 b = _mm256_set_ps (1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f);
    __m256 result = _mm256_add_ps (a, b);
    return 0;
  }" AVX_FOUND)
  if(AVX_FOUND)
    add_definitions(-DLITE_WITH_AVX)
  endif()
endif()
