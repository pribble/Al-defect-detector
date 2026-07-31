set(LITE_URL "http://paddle-inference-dist.bj.bcebos.com" CACHE STRING "inference download url")

function(lite_download_and_uncompress INSTALL_DIR URL FILENAME)
  set(options "")
  set(oneValueArgs MODEL_PATH)
  set(multiValueArgs "")
  cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(DEFINED args_MODEL_PATH)
    set(FILE_PATH ${args_MODEL_PATH}/${FILENAME})
    set(PREFIX ${INSTALL_DIR}/${args_MODEL_PATH})
    set(DOWNLOAD_DIR ${INSTALL_DIR}/${args_MODEL_PATH})
  else()
    set(FILE_PATH ${FILENAME})
    set(PREFIX ${INSTALL_DIR})
    set(DOWNLOAD_DIR ${INSTALL_DIR})
  endif()

  message(STATUS "Download inference test stuff: ${FILE_PATH}")
  string(REGEX REPLACE "[-%./]" "_" FILENAME_EX ${FILE_PATH})
  set(EXTERNAL_PROJECT_NAME "extern_lite_download_${FILENAME_EX}")
  set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")
  ExternalProject_Add(
            ${EXTERNAL_PROJECT_NAME}
            ${EXTERNAL_PROJECT_LOG_ARGS}
            PREFIX                ${PREFIX}
            DOWNLOAD_COMMAND      wget --no-check-certificate -q -O ${INSTALL_DIR}/${FILE_PATH} ${URL}/${FILE_PATH} && ${CMAKE_COMMAND} -E tar xzf ${INSTALL_DIR}/${FILE_PATH} && rm -f ${INSTALL_DIR}/${FILE_PATH}
            DOWNLOAD_DIR          ${DOWNLOAD_DIR}
            DOWNLOAD_NO_PROGRESS  1
            CONFIGURE_COMMAND     ""
            BUILD_COMMAND         ""
            UPDATE_COMMAND        ""
            INSTALL_COMMAND       ""
  )
endfunction()

function (lite_deps TARGET)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs DEPS ARM_DEPS INTEL_FPGA_DEPS ARGS)
  cmake_parse_arguments(lite_deps "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(deps ${lite_deps_DEPS})

  if(LITE_WITH_ARM)
    foreach(var ${lite_deps_ARM_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  if(LITE_WITH_INTEL_FPGA)
    foreach(var ${lite_deps_INTEL_FPGA_DEPS})
      set(deps ${deps} ${var})
    endforeach(var)
  endif()

  set(${TARGET} ${deps} PARENT_SCOPE)
endfunction()

add_custom_target(lite_compile_deps COMMAND echo 1)

set(offline_lib_registry_file "${PADDLE_BINARY_DIR}/lite_libs.txt")
file(WRITE ${offline_lib_registry_file} "")

function(lite_cc_library TARGET)
    set(options SHARED shared STATIC static MODULE module)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARM_DEPS INTEL_FPGA_DEPS EXCLUDE_COMPILE_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            INTEL_FPGA_DEPS ${args_INTEL_FPGA_DEPS}
            )

    if (args_SHARED OR ARGS_shared)
        cc_library(${TARGET} SRCS ${args_SRCS} DEPS ${deps} SHARED)
    elseif (args_MODULE OR ARGS_module)
        add_library(${TARGET} MODULE ${args_SRCS})
        add_dependencies(${TARGET} ${deps} ${args_DEPS})
    else()
        cc_library(${TARGET} SRCS ${args_SRCS} DEPS ${deps})
    endif()

    if(NOT WIN32)
      target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
    endif()
    if (args_SRCS AND NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()

    file(APPEND ${offline_lib_registry_file} "${TARGET}\n")
endfunction()

function(lite_cc_binary TARGET)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
        set(options " -g ")
    endif()
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARM_DEPS INTEL_FPGA_DEPS EXCLUDE_COMPILE_DEPS ARGS)
    cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(deps "")
    lite_deps(deps
            DEPS ${args_DEPS}
            ARM_DEPS ${args_ARM_DEPS}
            INTEL_FPGA_DEPS ${args_INTEL_FPGA_DEPS}
            )
    cc_binary(${TARGET} SRCS ${args_SRCS} DEPS ${deps})

    add_dependencies(${TARGET} bundle_full_api)

    if(NOT WIN32)
      target_link_libraries(${TARGET} ${PADDLE_BINARY_DIR}/libpaddle_api_full_bundled.a)
      target_compile_options(${TARGET} BEFORE PRIVATE -Wno-ignored-qualifiers)
    endif()

    if(LITE_WITH_INTEL_FPGA)
        target_link_libraries(${TARGET} ${intel_fpga_deps})
    endif()

    if (NOT APPLE AND NOT WIN32)
        if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            add_custom_command(TARGET ${TARGET} POST_BUILD
                    COMMAND "${CMAKE_STRIP}" -s
                    "${TARGET}"
                    COMMENT "Strip debug symbols done on final executable file.")
        endif()
    endif()
    if (NOT args_EXCLUDE_COMPILE_DEPS)
        add_dependencies(lite_compile_deps ${TARGET})
    endif()
endfunction()

function(bundle_static_library tgt_name bundled_tgt_name fake_target)
  list(APPEND static_libs ${tgt_name})
  add_dependencies(lite_compile_deps ${fake_target})

  function(_recursively_collect_dependencies input_target)
    set(_input_link_libraries LINK_LIBRARIES)
    get_target_property(_input_type ${input_target} TYPE)
    if (${_input_type} STREQUAL "INTERFACE_LIBRARY")
      set(_input_link_libraries INTERFACE_LINK_LIBRARIES)
    endif()
    get_target_property(public_dependencies ${input_target} ${_input_link_libraries})
    foreach(dependency IN LISTS public_dependencies)
      if(TARGET ${dependency})
        get_target_property(alias ${dependency} ALIASED_TARGET)
        if (TARGET ${alias})
          set(dependency ${alias})
        endif()
        get_target_property(_type ${dependency} TYPE)
        if (${_type} STREQUAL "STATIC_LIBRARY")
          list(APPEND static_libs ${dependency})
        endif()

        get_property(library_already_added
          GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency})
        if (NOT library_already_added)
          set_property(GLOBAL PROPERTY _${tgt_name}_static_bundle_${dependency} ON)
          _recursively_collect_dependencies(${dependency})
        endif()
      endif()
    endforeach()
    set(static_libs ${static_libs} PARENT_SCOPE)
  endfunction()

  _recursively_collect_dependencies(${tgt_name})

  list(REMOVE_DUPLICATES static_libs)

  set(bundled_tgt_full_name
    ${PADDLE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_tgt_name}${CMAKE_STATIC_LIBRARY_SUFFIX})

  file(WRITE ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar.in
    "CREATE ${bundled_tgt_full_name}\n" )

  foreach(tgt IN LISTS static_libs)
    file(APPEND ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar.in
      "ADDLIB $<TARGET_FILE:${tgt}>\n")
  endforeach()

  file(APPEND ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar.in "SAVE\n")
  file(APPEND ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar.in "END\n")

  file(GENERATE
    OUTPUT ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar
    INPUT ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar.in)

  set(ar_tool ${CMAKE_AR})
  if (CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(ar_tool ${CMAKE_CXX_COMPILER_AR})
  endif()

  add_custom_command(
    TARGET ${fake_target} PRE_BUILD
    COMMAND rm -f ${bundled_tgt_full_name}
    COMMAND ${ar_tool} -M < ${PADDLE_BINARY_DIR}/${bundled_tgt_name}.ar
    COMMENT "Bundling ${bundled_tgt_name}"
    DEPENDS ${tgt_name}
    VERBATIM)

  add_library(${bundled_tgt_name} STATIC IMPORTED)
  set_target_properties(${bundled_tgt_name}
    PROPERTIES
      IMPORTED_LOCATION ${bundled_tgt_full_name}
      INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${tgt_name},INTERFACE_INCLUDE_DIRECTORIES>)
  add_dependencies(${bundled_tgt_name} ${fake_target})

endfunction()

# No-op: tests disabled in tiny_publish mode
function(lite_cc_test TARGET)
endfunction()
