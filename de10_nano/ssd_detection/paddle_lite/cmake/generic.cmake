include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(NOT APPLE)
  find_package(Threads REQUIRED)
  link_libraries(${CMAKE_THREAD_LIBS_INIT})
  set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -pthread -ldl -lrt")
endif()

set_property(GLOBAL PROPERTY FLUID_MODULES "")

function(find_fluid_modules TARGET_NAME)
  get_filename_component(__target_path ${TARGET_NAME} ABSOLUTE)
  string(REGEX REPLACE "^${PADDLE_SOURCE_DIR}/" "" __target_path ${__target_path})
  string(FIND "${__target_path}" "lite" pos)
  if((pos GREATER 0) OR (pos EQUAL 0))
    get_property(fluid_modules GLOBAL PROPERTY FLUID_MODULES)
    set(fluid_modules ${fluid_modules} ${TARGET_NAME})
    set_property(GLOBAL PROPERTY FLUID_MODULES "${fluid_modules}")
  endif()
endfunction()

function(common_link TARGET_NAME)
endfunction()

function(merge_static_libs TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)

  foreach(lib ${libs})
    list(APPEND libs_deps ${${lib}_LIB_DEPENDS})
  endforeach()
  if(libs_deps)
    list(REMOVE_DUPLICATES libs_deps)
  endif()

  set(target_SRCS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c)
  set(target_DIR ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.dir)

  foreach(lib ${libs})
    set(objlistfile ${target_DIR}/${lib}.objlist)
    set(objdir ${target_DIR}/${lib}.objdir)

    add_custom_command(OUTPUT ${objdir}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${objdir}
      DEPENDS ${lib})

    add_custom_command(OUTPUT ${objlistfile}
      COMMAND ${CMAKE_AR} -x "$<TARGET_FILE:${lib}>"
      COMMAND ${CMAKE_AR} -t "$<TARGET_FILE:${lib}>" > ${objlistfile}
      DEPENDS ${lib} ${objdir}
      WORKING_DIRECTORY ${objdir})

    list(APPEND target_OBJS "${objlistfile}")
  endforeach()

  add_custom_command(OUTPUT ${target_SRCS}
    COMMAND ${CMAKE_COMMAND} -E touch ${target_SRCS}
    DEPENDS ${libs} ${target_OBJS})

  file(WRITE ${target_SRCS} "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")
  add_library(${TARGET_NAME} STATIC ${target_SRCS})
  target_link_libraries(${TARGET_NAME} ${libs_deps})

  set(target_LIBNAME "$<TARGET_FILE:${TARGET_NAME}>")

  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_AR} crs ${target_LIBNAME} `find ${target_DIR} -name '*.o'`
      COMMAND ${CMAKE_RANLIB} ${target_LIBNAME}
      WORKING_DIRECTORY ${target_DIR})
endfunction()

function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(cc_library_SRCS)
    if(cc_library_SHARED OR cc_library_shared)
      add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
      find_fluid_modules(${TARGET_NAME})
    endif()

    if(cc_library_DEPS)
      if("${cc_library_DEPS};" MATCHES "fbs_headers;")
        list(REMOVE_ITEM cc_library_DEPS fbs_headers)
        add_dependencies(${TARGET_NAME} fbs_headers)
      endif()
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
      add_dependencies(${TARGET_NAME} ${cc_library_DEPS})
      common_link(${TARGET_NAME})
    endif()

    set(full_path_src "")
    foreach(source_file ${cc_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cc_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
      if(${source_file} MATCHES ${PADDLE_SOURCE_DIR} AND NOT ${source_file} MATCHES "framework.pb.cc")
        list(APPEND full_path_src ${source_file})
      elseif( NOT ${source_file} MATCHES "framework.pb.cc")
        list(APPEND full_path_src ${CMAKE_CURRENT_SOURCE_DIR}/${source_file})
      endif()
    endforeach()
    set(__lite_cc_files ${__lite_cc_files} ${full_path_src} CACHE INTERNAL "")
  else()
    if(cc_library_DEPS)
      merge_static_libs(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL_ERROR "Please specify source files or libraries in cc_library(${TARGET_NAME} ...).")
    endif()
  endif()
endfunction()

function(cc_binary TARGET_NAME)
  set(options "")
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_binary "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  add_executable(${TARGET_NAME} ${cc_binary_SRCS})
  if(cc_binary_DEPS)
    target_link_libraries(${TARGET_NAME} ${cc_binary_DEPS})
    add_dependencies(${TARGET_NAME} ${cc_binary_DEPS})
    common_link(${TARGET_NAME})
  endif()
  get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
  target_link_libraries(${TARGET_NAME} ${os_dependency_modules})
  find_fluid_modules(${TARGET_NAME})
endfunction(cc_binary)
