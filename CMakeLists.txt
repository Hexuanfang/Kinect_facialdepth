cmake_minimum_required(VERSION 2.8.3)
project(xbox_tracker)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

OPTION(ENABLE_CXX11 "Enable C++11 support" ON)

#Suppress deprecated strings and unused result warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -Wno-unused-result")

set(Boost_LIBRARY_DIR /usr/lib/x86_64-linux-gnu)

IF(ENABLE_CXX11)
  INCLUDE(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  IF(COMPILER_SUPPORTS_CXX11)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  ELSEIF(COMPILER_SUPPORTS_CXX0X)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  ELSE()
    MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  ENDIF()
ENDIF(ENABLE_CXX11)

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  #Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()


find_package(PkgConfig)
find_package(OpenCV REQUIRED)
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
find_package(PythonLibs REQUIRED)
find_package(Boost COMPONENTS system thread REQUIRED)

IF(OpenCV_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})
ENDIF(OpenCV_FOUND)

SET(LIBRARIES
${OpenCV_LIBS}
#${Matlab_LIBRARIES}
  )

include_directories(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})

# matlab
#set(MATLAB_INCLUDE_DIR "/usr/local/MATLAB/R2015a/extern/include")
#set(MATLAB_LIBRARY_DIR "/usr/local/MATLAB/R2015a/bin/glnxa64")
#set(MATLAB_LIBRARIES
#  ${MATLAB_LIBRARY_DIR}/libeng.so
#  ${MATLAB_LIBRARY_DIR}/libmx.so
#  ${MATLAB_LIBRARY_DIR}/libmex.so
#  )



INCLUDE_DIRECTORIES(
                    ${OpenCV_INCLUDE_DIRS}
                    ${PYTHON_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIRS}
#                    ${MATLAB_INCLUDE_DIR}
                    )

add_definitions(${EIGEN_DEFINITIONS})

# Print some opencv messages
message(STATUS "\nOpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: ${OpenCV_LIBS}")

message(STATUS  "Boost Include Directory: ${Boost_INCLUDE_DIRS}")
message(STATUS  "\n\nBoost_LIBRARIES: \n\n${Boost_LIBRARIES}")
message(STATUS  "\n\nBoost_LIBRARY_DIR: \n\n${Boost_LIBRARY_DIR}")

link_directories(${OpenCV_LIBS})

#Declare the executable target built from your sources
add_executable(xbox_tracker xbox_tracker.cpp)
add_executable(listener xbox_listener.cpp)
add_executable(ros_listener ros_listener.cpp)  
add_executable(kinect_calibration kinect_calibration.cpp)
add_executable(kinect_calibration_depth kinect_calibration_depth.cpp)
add_executable(stereo_calibration stereo_calibration.cpp)
  #as a check

# Link your application with OpenCV libraries
target_link_libraries(xbox_tracker ${OpenCV_LIBS})
target_link_libraries(xbox_tracker ${OpenCV_LIBRARIES})
target_link_libraries(listener ${OpenCV_LIBRARIES})
target_link_libraries(listener ${Boost_LIBRARIES})
target_link_libraries(kinect_calibration ${OpenCV_LIBS})
target_link_libraries(kinect_calibration ${OpenCV_LIBRARIES})
target_link_libraries(kinect_calibration_depth ${OpenCV_LIBS})
target_link_libraries(kinect_calibration_depth ${OpenCV_LIBRARIES})
target_link_libraries(stereo_calibration ${OpenCV_LIBS})
target_link_libraries(stereo_calibration ${OpenCV_LIBRARIES})
