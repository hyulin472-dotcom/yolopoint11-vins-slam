cmake_minimum_required(VERSION 3.5)
project(vins)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(VINS_ENABLE_CUDA "Build the optional CUDA keypoint postprocessor" ON)
include(CheckLanguage)
if(VINS_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
  else()
    message(WARNING "CUDA compiler not found; building with CPU postprocessing")
    set(VINS_ENABLE_CUDA OFF)
  endif()
endif()
set(VINS_CUDA_ARCHITECTURES "" CACHE STRING
  "CUDA architectures to build, for example 86 or 89; empty uses nvcc defaults")
if(VINS_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${VINS_CUDA_ARCHITECTURES}")
elseif(POLICY CMP0104)
  cmake_policy(SET CMP0104 OLD)
endif()
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
  $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcpputils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(camera_models REQUIRED)
find_package(image_transport REQUIRED)


find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
get_target_property(CERES_TARGET_INCLUDE_DIRS
  Ceres::ceres INTERFACE_INCLUDE_DIRECTORIES)
list(GET CERES_TARGET_INCLUDE_DIRS 0 CERES_PRIMARY_INCLUDE_DIR)
set(CERES_HEADER_SHIM "${CMAKE_CURRENT_BINARY_DIR}/ceres_include")
file(MAKE_DIRECTORY "${CERES_HEADER_SHIM}")
file(CREATE_LINK "${CERES_PRIMARY_INCLUDE_DIR}/ceres"
  "${CERES_HEADER_SHIM}/ceres" SYMBOLIC COPY_ON_ERROR)
include_directories(BEFORE "${CERES_HEADER_SHIM}")

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime C/C++ SDK root")
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS "${ONNXRUNTIME_ROOT}/include")
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64")
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime C++ SDK not found. Set ONNXRUNTIME_ROOT or pass "
    "-DONNXRUNTIME_ROOT=/path/to/onnxruntime.")
endif()
get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
set(VINS_BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR}")
message(STATUS "Using ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Using ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "CUDA postprocessing: ${VINS_ENABLE_CUDA}")

# include camera_models
include_directories("../camera_models/include")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
)
set(VINS_CUDA_POSTPROCESS_SOURCE src/featureTracker/yolopoint_cuda_postprocess_stub.cpp)
if(VINS_ENABLE_CUDA)
  set(VINS_CUDA_POSTPROCESS_SOURCE src/featureTracker/yolopoint_cuda_postprocess.cu)
endif()

add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/online_dense_mapper.cpp
    src/utility/CameraPoseVisualization.cpp
    src/loop_closure/independent_loop_closure.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
    ${VINS_CUDA_POSTPROCESS_SOURCE}
    src/featureTracker/yolopoint_lightglue_feature_tracker.cpp)
target_include_directories(vins_lib PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
target_compile_definitions(vins_lib PUBLIC VINS_HAS_CUDA=$<BOOL:${VINS_ENABLE_CUDA}>)
target_link_libraries(vins_lib ${OpenCV_LIBS} Ceres::ceres ${ONNXRUNTIME_LIBRARY} ${CMAKE_DL_LIBS})
if(VINS_ENABLE_CUDA)
  set_target_properties(vins_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
set_target_properties(vins_lib PROPERTIES
    BUILD_RPATH "${VINS_BUILD_RPATH}"
    INSTALL_RPATH "")

ament_target_dependencies(vins_lib rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge camera_models image_transport)

add_executable(vins_node src/rosNodeTest.cpp)
ament_target_dependencies(vins_node rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge camera_models image_transport)
target_link_libraries(vins_node vins_lib) 
set_target_properties(vins_node PROPERTIES
    BUILD_RPATH "${VINS_BUILD_RPATH}"
    INSTALL_RPATH "")

add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
ament_target_dependencies(kitti_odom_test rclcpp rcpputils std_msgs visualization_msgs geometry_msgs nav_msgs tf2 tf2_ros cv_bridge camera_models image_transport)
target_link_libraries(kitti_odom_test vins_lib) 
set_target_properties(kitti_odom_test PROPERTIES
    BUILD_RPATH "${VINS_BUILD_RPATH}"
    INSTALL_RPATH "")

add_executable(foundation_stereo_ort_benchmark ../tools/foundation_stereo_ort_benchmark.cpp)
target_include_directories(foundation_stereo_ort_benchmark PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(foundation_stereo_ort_benchmark ${ONNXRUNTIME_LIBRARY})
set_target_properties(foundation_stereo_ort_benchmark PROPERTIES
    BUILD_RPATH "${VINS_BUILD_RPATH}"
    INSTALL_RPATH "")

add_executable(yolopoint_lightglue_tracker_smoke ../tools/yolopoint_lightglue_tracker_smoke.cpp)
target_link_libraries(yolopoint_lightglue_tracker_smoke vins_lib)
set_target_properties(yolopoint_lightglue_tracker_smoke PROPERTIES
    BUILD_RPATH "${VINS_BUILD_RPATH}"
    INSTALL_RPATH "")

add_executable(euroc_stereo_diagnostic ../tools/euroc_stereo_diagnostic.cpp)
target_link_libraries(euroc_stereo_diagnostic vins_lib)
set_target_properties(euroc_stereo_diagnostic PROPERTIES
    BUILD_RPATH "${VINS_BUILD_RPATH}"
    INSTALL_RPATH "")

# added from DEBUG

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch/
)

# Install config files (includes .rviz and YAML configs)
install(
  DIRECTORY ../config/
  DESTINATION share/${PROJECT_NAME}/config/
)

install(
  DIRECTORY ../onxx/yolopointv11_lightglue
  DESTINATION share/${PROJECT_NAME}/onxx
  PATTERN "trt_cache" EXCLUDE
)

install(
  DIRECTORY ../onxx/fast_foundationstereo
  DESTINATION share/${PROJECT_NAME}/onxx
  PATTERN "trt_cache" EXCLUDE
)

# Install nodes
install(
  TARGETS vins_node kitti_odom_test
  DESTINATION lib/${PROJECT_NAME}
)



# Install binaries
#install(TARGETS vins_node kitti_odom_test vins_lib
#    ARCHIVE DESTINATION lib
#    LIBRARY DESTINATION lib
#    RUNTIME DESTINATION bin
#)

# Install launch files
#install(DIRECTORY
#    launch
#    # config
#    # rviz
#    DESTINATION share/${PROJECT_NAME}
#    )


ament_package()
