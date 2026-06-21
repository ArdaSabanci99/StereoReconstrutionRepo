cmake_minimum_required(VERSION 3.16)
project(StereoReconstructionNew CXX)
set(CMAKE_CXX_STANDARD 17)

# ── Library search path (same layout as exercises) ──────────────────────────
# Project/stereo-reconstruction-new/ → ../Libs = Project/Libs/
set(LIBRARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Libs" CACHE PATH "Path to Libs/ folder")

# ── OpenCV ──────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── Eigen ───────────────────────────────────────────────────────────────────
# Prefer imported target Eigen3::Eigen (works with vcpkg + find_package).
# Fallback: manually locate header-only Eigen in Libs/ or vcpkg include/.
find_package(Eigen3 QUIET)
if(Eigen3_FOUND)
    message(STATUS "Eigen3: ${Eigen3_VERSION} (imported target Eigen3::Eigen)")
    set(EIGEN3_INCLUDE_DIR "")
    set(EIGEN3_TARGET Eigen3::Eigen)
else()
    # Manual fallback paths (header-only, no cmake config needed)
    set(_EIGEN_CANDIDATES
        "${LIBRARY_DIR}/Eigen/include/eigen3"
        "C:/vcpkg/installed/x64-windows/include/eigen3"
        "C:/vcpkg/installed/x64-windows/include"
    )
    set(EIGEN3_INCLUDE_DIR "")
    set(EIGEN3_TARGET "")
    foreach(_p ${_EIGEN_CANDIDATES})
        if(EXISTS "${_p}/Eigen/Core")
            set(EIGEN3_INCLUDE_DIR "${_p}")
            message(STATUS "Eigen3: found at ${_p} (manual)")
            break()
        endif()
    endforeach()
    if(NOT EIGEN3_INCLUDE_DIR)
        message(FATAL_ERROR
            "Eigen3 not found.\n"
            "Fix: run  C:\\vcpkg\\vcpkg.exe install eigen3:x64-windows\n"
            "     then re-run cmake.")
    endif()
endif()

# ── Ceres + glog (for ICP) ──────────────────────────────────────────────────
# Only look in the local Libs/ folder to avoid pulling in vcpkg's Ceres
# (which requires gflags that may not be installed). ICP is optional.
if(EXISTS "${LIBRARY_DIR}/Ceres/CMake/CeresConfig.cmake")
    find_package(Ceres QUIET HINTS "${LIBRARY_DIR}/Ceres/CMake" NO_DEFAULT_PATH)
else()
    set(Ceres_FOUND FALSE)
endif()
if(NOT Ceres_FOUND)
    message(STATUS "Ceres not found — icp_fuse target will not be built")
endif()

# ── FLANN (for ICP nearest-neighbour) ───────────────────────────────────────
set(FLANN_INCLUDE_DIR "${LIBRARY_DIR}/Flann-1.8.4/include" CACHE PATH "")
set(FLANN_LIBRARY     "${LIBRARY_DIR}/Flann-1.8.4/lib/Release/flann_cpp_s.lib" CACHE FILEPATH "")

# ── Include dirs shared by all targets ──────────────────────────────────────
include_directories(
    include/
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${FLANN_INCLUDE_DIR}
)

# ── Helper: add an executable linked against common libs ────────────────────
function(add_stereo_exe TARGET)
    add_executable(${TARGET} ${ARGN})
    target_link_libraries(${TARGET} ${OpenCV_LIBS})
    # Link Eigen imported target if found via find_package
    if(EIGEN3_TARGET)
        target_link_libraries(${TARGET} ${EIGEN3_TARGET})
    endif()
    if(WIN32)
        target_compile_definitions(${TARGET} PRIVATE _USE_MATH_DEFINES NOMINMAX)
    endif()
endfunction()

# ── Shared sources used in multiple targets ──────────────────────────────────
set(COMMON_SRC
    src/DataLoader.cpp
    src/utils.cpp
)

# ── 1. Sparse matching ───────────────────────────────────────────────────────
add_stereo_exe(sparse_matching
    ${COMMON_SRC}
    src/sparse_matching.cpp
)

# ── 2. Rectification ────────────────────────────────────────────────────────
add_stereo_exe(rectification
    ${COMMON_SRC}
    src/sparse_matching.cpp
    src/rectification.cpp
)
target_compile_definitions(rectification PRIVATE BUILDING_RECTIFICATION)

# ── 3. Dense matching ────────────────────────────────────────────────────────
add_stereo_exe(matching
    ${COMMON_SRC}
    src/matching.cpp
)

# ── 4. Depth & point cloud ──────────────────────────────────────────────────
add_stereo_exe(depth
    ${COMMON_SRC}
    src/depth.cpp
)

# ── 5. Evaluation ───────────────────────────────────────────────────────────
add_stereo_exe(evaluation
    ${COMMON_SRC}
    src/matching.cpp
    src/evaluation.cpp
)
# matching.cpp has its own main guarded by #ifndef PIPELINE_BUILD;
# suppress it so evaluation.cpp's main is the only one.
target_compile_definitions(evaluation PRIVATE PIPELINE_BUILD)

# ── 5b. DTU point-cloud evaluation (standalone) ──────────────────────────────
add_stereo_exe(dtu_eval
    src/dtu_eval.cpp
)

# ── 6. Full pipeline ─────────────────────────────────────────────────────────
add_stereo_exe(pipeline
    ${COMMON_SRC}
    src/sparse_matching.cpp
    src/rectification.cpp
    src/matching.cpp
    src/depth.cpp
    src/evaluation.cpp
    src/dtu_eval.cpp
    src/pipeline.cpp
)
# PIPELINE_BUILD suppresses matching.cpp's main; SUPPRESS_EVAL_MAIN suppresses
# evaluation.cpp's main — pipeline.cpp provides the only main for this target.
target_compile_definitions(pipeline PRIVATE PIPELINE_BUILD SUPPRESS_EVAL_MAIN)

# ── 7. ICP fusion (requires Ceres) ──────────────────────────────────────────
if(Ceres_FOUND)
    add_stereo_exe(icp_fuse
        ${COMMON_SRC}
        src/depth.cpp
        src/icp.cpp
    )
    target_link_libraries(icp_fuse Ceres::ceres ${FLANN_LIBRARY})
    if(WIN32)
        target_link_libraries(icp_fuse "${LIBRARY_DIR}/glog-lib/lib/Release/glog.lib")
    endif()
endif()
