cmake_minimum_required(VERSION 2.8)
project(KITTICornerDetection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC  -lm -Wl,--no-as-needed -DLINUX -g -Wall \
-std=c++11 -Wunused-variable -Wwrite-strings -Wsign-compare -fpermissive ")
set(CMAKE_BUILD_TYPE "Release")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(OpenCV_LIBS ${OpenCV_LIBRARIES})



INCLUDE_DIRECTORIES(./)
AUX_SOURCE_DIRECTORY(./ gSOURCE_FILES_)

add_library(${PROJECT_NAME} SHARED 
	    ${gSOURCE_FILES_}
  	    )    
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(FILES
  ChessboradStruct.h
  corealgmatlab.h
  CornerDetAC.h
  HeaderCB.h
  DESTINATION include
)
