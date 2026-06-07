cmake_minimum_required(VERSION 3.16)
project(StereoReconstruction VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Library paths (mirrors Exercise 5 layout) ─────────────────────────────
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to shared Libs folder")

set(Eigen3_DIR   ${LIBRARY_DIR}/Eigen/share/eigen3/cmake          CACHE PATH "Eigen cmake dir")
set(glog_DIR     ${LIBRARY_DIR}/glog-lib/lib/cmake/glog            CACHE PATH "glog cmake dir")
set(Ceres_DIR    ${LIBRARY_DIR}/Ceres/CMake                        CACHE PATH "Ceres cmake dir")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4/                  CACHE PATH "FLANN include dir")

# ── OpenCV ────────────────────────────────────────────────────────────────
if(WIN32 AND NOT DEFINED OpenCV_DIR)
  foreach(_candidate
      "C:/opencv/build"
      "C:/opencv/build/x64/vc16/lib"
      "C:/tools/opencv/build"
      "$ENV{USERPROFILE}/opencv/build")
    if(EXISTS "${_candidate}/OpenCVConfig.cmake" OR
       EXISTS "${_candidate}/opencv-config.cmake")
      set(OpenCV_DIR "${_candidate}")
      message(STATUS "Auto-detected OpenCV at ${OpenCV_DIR}")
      break()
    endif()
  endforeach()
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ── Eigen ─────────────────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")

# ── Ceres + glog ──────────────────────────────────────────────────────────
find_package(Ceres REQUIRED)
find_package(glog  REQUIRED)
get_target_property(GLOG_DLL_PATH_DEBUG   glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)
add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

# ── Open3D (optional) ─────────────────────────────────────────────────────
find_package(Open3D QUIET)

# ── Global include directories ────────────────────────────────────────────
include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    ${Flann_INCLUDE_DIR}
)

# ── Individual executables ────────────────────────────────────────────────

add_executable(sparse_matching src/sparse_matching.cpp src/utils.cpp)
target_link_libraries(sparse_matching ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(rectification src/rectification.cpp src/sparse_matching.cpp src/utils.cpp)
target_link_libraries(rectification ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(matching src/matching.cpp src/utils.cpp)
target_link_libraries(matching ${OpenCV_LIBS})

add_executable(depth src/depth.cpp src/utils.cpp)
target_link_libraries(depth ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(evaluation src/evaluation.cpp src/utils.cpp)
target_link_libraries(evaluation ${OpenCV_LIBS} Eigen3::Eigen)

add_executable(icp src/icp.cpp src/depth.cpp src/utils.cpp)
target_link_libraries(icp ${OpenCV_LIBS} Eigen3::Eigen ceres)
target_include_directories(icp PRIVATE ${Flann_INCLUDE_DIR})

# ── Pipeline (shared object library) ─────────────────────────────────────
add_library(stereo_lib OBJECT
    src/sparse_matching.cpp
    src/rectification.cpp
    src/matching.cpp
    src/depth.cpp
    src/utils.cpp
)
target_compile_definitions(stereo_lib PRIVATE PIPELINE_BUILD)
target_include_directories(stereo_lib PRIVATE include ${OpenCV_INCLUDE_DIRS}
                                               ${EIGEN3_INCLUDE_DIR} ${Flann_INCLUDE_DIR})

add_executable(pipeline src/pipeline.cpp)
target_link_libraries(pipeline stereo_lib ${OpenCV_LIBS} Eigen3::Eigen)
target_compile_definitions(pipeline PRIVATE PIPELINE_BUILD)

# ── Mesh (Open3D, optional) ───────────────────────────────────────────────
if(Open3D_FOUND)
    message(STATUS "Open3D found — mesh reconstruction enabled")
    add_executable(mesh src/mesh.cpp src/utils.cpp)
    target_link_libraries(mesh ${OpenCV_LIBS} Eigen3::Eigen Open3D::Open3D)
else()
    message(STATUS "Open3D not found — mesh step will be skipped")
endif()

# ── Windows: copy DLLs ────────────────────────────────────────────────────
if(WIN32)
    foreach(_target sparse_matching rectification matching depth evaluation icp pipeline)
        if(TARGET ${_target})
            add_custom_command(TARGET ${_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${GLOG_DLL_PATH_DEBUG}" $<TARGET_FILE_DIR:${_target}>)
            add_custom_command(TARGET ${_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${GLOG_DLL_PATH_RELEASE}" $<TARGET_FILE_DIR:${_target}>)
        endif()
    endforeach()
endif()
