# Reference:

#The name of the project
PROJECT(GPUDistToTrianglesSimple)

FIND_PACKAGE(CUDA)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(MYLIB_DIR 		"~/mylib" 	CACHE PATH "The path to MyLib.")	
set(3RDPARTY_DIR 	"~/lib" 	CACHE PATH "The path to 3rd party libraries.")	

include_directories(
	${MYLIB_DIR}/include		
	${3RDPARTY_DIR}/include	
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CUDA_SDK_ROOT_DIR}/common/inc
	../GPUDistLib/	
)

link_directories(
	${MYLIB_DIR}/lib		
	${3RDPARTY_DIR}/lib
	${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32
	${CUDA_SDK_ROOT_DIR}/common/lib/Win32
)

ADD_EXECUTABLE(${PROJECT_NAME}
	GPUDistToTrianglesSimple_main.cpp
)

target_link_libraries(${PROJECT_NAME}
	GPUDistLib
	)

# link OpenGL-related libraries
if(DEFINED WIN32)
	target_link_libraries(${PROJECT_NAME}
	)
else()
	target_link_libraries(${PROJECT_NAME}
	)
endif()

# the needed libraries
if(DEFINED WIN32)	
	target_link_libraries(${PROJECT_NAME}
		debug	libopt_d	optimized	libopt_r
		debug	lib3ds-120sd	optimized	lib3ds-120sd
	)
else()
	target_link_libraries(${PROJECT_NAME}
		libopt
		lib3ds-120s 
	)
endif()

install(
	TARGETS ${PROJECT_NAME}
	RUNTIME	DESTINATION bin
	ARCHIVE	DESTINATION	lib
	)	
	
add_dependencies(${PROJECT_NAME}	GPUDistLib)	

#
# $Log$
#