cmake_minimum_required(VERSION 3.5)
project(turtlebot_simulation)

find_package(ament_cmake REQUIRED)
find_package(kobuki_description REQUIRED)
find_package(swiftpro_description REQUIRED)
find_package(turtlebot_description REQUIRED)
find_package(stonefish_ros2 REQUIRED)

install(DIRECTORY launch resources scenarios scripts
  DESTINATION share/${PROJECT_NAME}
)
install(
  PROGRAMS 
    scripts/swiftpro_controller.py
    scripts/diff_drive_odometry.py
    scripts/diff_drive_controller.py
    scripts/odometry_to_tf.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()