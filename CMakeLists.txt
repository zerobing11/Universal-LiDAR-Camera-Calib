cmake_minimum_required( VERSION 2.8 )
project( camera_calibration )

set( CMAKE_BUILD_TYPE "Debug" )
add_compile_options(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
set(OpenCV_DIR "/usr/local/OpenCV345/share/OpenCV")
set(OpenCV_DIR "/usr/local/share/OpenCV")

set(YAML_CPP_DIR "${CMAKE_SOURCE_DIR}/env/yaml-cpp-install/lib/cmake/yaml-cpp")
find_package(YAML_CPP REQUIRED)
set(YAML_LIB "${CMAKE_SOURCE_DIR}/env/yaml-cpp-install/lib/libyaml-cpp.so")
MESSAGE(STATUS "YAML_CPP_INCLUDE_DIR = ${YAML_CPP_INCLUDE_DIR}")
MESSAGE(STATUS "YAML_CPP_LIBRARIES = ${YAML_CPP_LIBRARIES}")

find_package( OpenCV REQUIRED )
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV source directory: ${OpenCV_SOURCE_DIR}")

find_package(Ceres REQUIRED)

find_package(PCL REQUIRED)
message(STATUS "PCL include directories: ${PCL_INCLUDE_DIRS}")

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs cv_bridge sensor_msgs pcl_ros message_generation)
catkin_package()

set(SQPNP_INSTALL_ROOT "${PROJECT_SOURCE_DIR}/env/sqpnp-install")
set(SQPNP_LIB "${SQPNP_INSTALL_ROOT}/lib/libSQPnP.so")

set(CORNER_DET_INSTALL_ROOT "${PROJECT_SOURCE_DIR}/env/corner-detection-install")
set(CORNER_DET_LIB "${CORNER_DET_INSTALL_ROOT}/lib/libKITTICornerDetection.so")

include_directories(
		    ${PROJECT_SOURCE_DIR}/include
			${PROJECT_SOURCE_DIR}/src
		    ${OpenCV_INCLUDE_DIRS}
		    ${catkin_INCLUDE_DIRS} 
		    "/usr/include/eigen3"
		    ${CERES_INCLUDE_DIRS}
		    ${PCL_INCLUDE_DIRS}
		    ${YAML_CPP_INCLUDE_DIR}
		    ${SQPNP_INSTALL_ROOT}/include
		    ${CORNER_DET_INSTALL_ROOT}/include
		    )

set(ENV_LIBS
		${YAML_LIB}
		${OpenCV_LIBS}
		${CERES_LIBRARIES}
		${catkin_LIBRARIES}
		${PCL_LIBRARIES}
		${CORNER_DET_LIB}
		${SQPNP_LIB}
)
add_subdirectory(src/lidar_sort)
add_subdirectory(src/otsu_filter)
add_subdirectory(src/coarse2fine)
add_subdirectory(src/lidar_corners_detect)
add_subdirectory(src/camera)
add_subdirectory(src/refine)



add_executable( main_MSR src/main_MSR.cpp src/config_helper.cpp)
target_link_libraries( main_MSR
		lidar_sort
		otsu_filter
		coarse2fine
		lidar_corners_detect
		camera
		${ENV_LIBS}
)


add_executable( main_SSR src/main_SSR.cpp src/config_helper.cpp)
target_link_libraries( main_SSR
		lidar_sort
		otsu_filter
		coarse2fine
		lidar_corners_detect
		camera
		refine
		${ENV_LIBS}
)
