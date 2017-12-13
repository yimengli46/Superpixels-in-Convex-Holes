# CMakeLists.txt for VOCUS 2 saliency system (Frintrop et al. CVPR 2015)
#
# see http://www.cmake.org/ for a documentation about CMake
#
# compile your programm using the following commands:
# $ cmake CMakeLists.txt
# $ make

cmake_minimum_required(VERSION 2.8)

project(superpixel)

# CGAL and its components
find_package( CGAL REQUIRED COMPONENTS CORE)
include( ${CGAL_USE_FILE} )

message("CGAL_LIBRARIES: " ${CGAL_LIBRARIES})
message("CGAL_3RD_PARTY_LIBRARIES: " ${CGAL_3RD_PARTY_LIBRARIES}) 

# choose if you want to compile with or without debugging flags,
# uncomment only one of the following two commands at a time
#set(CMAKE_BUILD_TYPE release)
set(CMAKE_BUILD_TYPE debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

find_package(OpenCV REQUIRED HINTS "/usr/local")
message("OpenCV_INCLUDE_DIR: " ${OpenCV_INCLUDE_DIRS})

find_package(Eigen3) #(requires 3.2.0 or greater)
#if (EIGEN3_FOUND)
#  include( ${EIGEN3_USE_FILE} )
#endif()

include( CGAL_CreateSingleSourceCGALProgram)
#create_single_source_cgal_program( "convex_hull_segmentation.cpp lsd_cmd.cpp lsd.c shape_detection.cpp" )
add_executable(convex_hull_segmentation convex_hull_segmentation.cpp shape_detection.cpp voronoi_diagram.cpp lsd_cmd.cpp lsd.c)
#create_single_source_cgal_program( "convex_hull_segmentation.cpp")
#add_executable(convex_hull_segmentation convex_hull_segmentation.cpp lsd_cmd.cpp lsd.c shape_detection.cpp)
target_link_libraries(convex_hull_segmentation ${OpenCV_LIBS} ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})

#add_executable(cannyEdgeDetection canny_edge.cpp)
#target_link_libraries(cannyEdgeDetection ${OpenCV_LIBS})

#include( CGAL_CreateSingleSourceCGALProgram )
#create_single_source_cgal_program( "shape_detection.cpp" )
#target_link_libraries(shape_detection ${OpenCV_LIBS})

#create_single_source_cgal_program( "voronoi_diagram.cpp" )
#target_link_libraries(voronoi_diagram ${OpenCV_LIBS})

