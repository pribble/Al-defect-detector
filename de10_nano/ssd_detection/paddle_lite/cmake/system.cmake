IF(WIN32)
    SET(HOST_SYSTEM "win32")
ELSE(WIN32)
    IF(APPLE)
        SET(HOST_SYSTEM "macosx")
    ELSE(APPLE)
        IF(EXISTS "/etc/issue")
            FILE(READ "/etc/issue" LINUX_ISSUE)
            IF(LINUX_ISSUE MATCHES "CentOS")
                SET(HOST_SYSTEM "centos")
            ELSEIF(LINUX_ISSUE MATCHES "Debian")
                SET(HOST_SYSTEM "debian")
            ELSEIF(LINUX_ISSUE MATCHES "Ubuntu")
                SET(HOST_SYSTEM "ubuntu")
            ELSEIF(LINUX_ISSUE MATCHES "Red Hat")
                SET(HOST_SYSTEM "redhat")
            ELSEIF(LINUX_ISSUE MATCHES "Fedora")
                SET(HOST_SYSTEM "fedora")
            ENDIF()
        ENDIF()
        IF(NOT HOST_SYSTEM)
            SET(HOST_SYSTEM ${CMAKE_SYSTEM_NAME})
        ENDIF()
    ENDIF(APPLE)
ENDIF(WIN32)

CMAKE_HOST_SYSTEM_INFORMATION(RESULT CPU_CORES QUERY NUMBER_OF_LOGICAL_CORES)
MARK_AS_ADVANCED(HOST_SYSTEM CPU_CORES)
MESSAGE(STATUS "Found Paddle host system: ${HOST_SYSTEM}")
MESSAGE(STATUS "Found Paddle host system's CPU: ${CPU_CORES} cores")

SET(EXTERNAL_PROJECT_LOG_ARGS
    LOG_DOWNLOAD    0
    LOG_UPDATE      1
    LOG_CONFIGURE   1
    LOG_BUILD       0
    LOG_TEST        1
    LOG_INSTALL     0
)
