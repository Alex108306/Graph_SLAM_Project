cmake_minimum_required(VERSION 3.8)
project(swiftpro_description)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(swiftpro_rviz_node src/swiftpro_rviz_node.cpp)
ament_target_dependencies(swiftpro_rviz_node rclcpp sensor_msgs)

install(
  TARGETS swiftpro_rviz_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY 
  resources 
  rviz 
  urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()