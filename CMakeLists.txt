cmake_minimum_required(VERSION 3.20)
project(aruco_orbslam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d objdetect)
find_package(ORB_SLAM3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)

add_executable(slam src/main.cpp)
# libORB_SLAM3 is built with -march=native, and the instruction set changes
# Eigen's alignment and so the layout of ORB-SLAM3's classes (e.g. the tracker's
# trajectory lists), so code reaching into them must be compiled the same way
target_compile_options(slam PRIVATE -march=native)
target_link_libraries(slam
  ament_index_cpp::ament_index_cpp
  cv_bridge::cv_bridge
  Eigen3::Eigen
  ${geometry_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${OpenCV_LIBS}
  ORB_SLAM3::ORB_SLAM3
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  tf2_eigen::tf2_eigen
)

install(TARGETS slam
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
