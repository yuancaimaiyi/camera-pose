cmake_minimum_required( VERSION 2.8 )
project( ceres_campose )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 添加cmake模块以使用ceres库
#list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set(Ceres_DIR /usr/local/ceres/lib/cmake/Ceres)
include_directories( "/usr/include/eigen3" )
# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )
set(OpenCV_DIR /usr/local/opencv3.4.3/share/OpenCV)
# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )
add_executable( campose main.cpp )
# 与Ceres和OpenCV链接
target_link_libraries( campose ${CERES_LIBRARIES} ${OpenCV_LIBS} )
