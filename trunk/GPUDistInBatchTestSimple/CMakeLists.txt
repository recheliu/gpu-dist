# Reference:

#The name of the project
PROJECT(GPUDistInBatchTestSimple)

FIND_PACKAGE(CUDA)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(MYLIB_DIR 		"~/mylib" 	CACHE PATH "The path to MyLib.")	
set(3RDPARTY_DIR 	"~/lib" 	CACHE PATH "The path to 3rd party libraries.")	
set(IS_X64 			OFF 		CACHE BOOL "Is the platform x64?")	# TEST-ADD

include_directories(
	${MY_LIB_DIR}/include
	${3RDPARTY_DIR}/include	
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CUDA_SDK_ROOT_DIR}/common/inc
	../GPUDistLib/	
)

link_directories(
	${MY_LIB_DIR}/lib
	${3RDPARTY_DIR}/lib
	${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32
	${CUDA_SDK_ROOT_DIR}/common/lib/Win32
)

ADD_EXECUTABLE(${PROJECT_NAME}
	GPUDistInBatchTestSimple_main.cpp
)

# link CUDPP
# if(DEFINED WIN32)	
#	target_link_libraries(${PROJECT_NAME}
#		debug	cudpp32d	optimized	cudpp32
#	)
# else()
# TEST-MOD-FROM:
 	# target_link_libraries(${PROJECT_NAME}
		# cudpp32
 	# )
# TEST-MOD-TO:
if( IS_X64 )
 	target_link_libraries(${PROJECT_NAME}
		cudpp64
 	)
else()
 	target_link_libraries(${PROJECT_NAME}
		cudpp32
 	)
endif()
# TEST-MOD-END
# endif()
	
# the needed libraries
if(DEFINED WIN32)	
	target_link_libraries(${PROJECT_NAME}
		debug	libopt_d	optimized	libopt_r
	)
else()
	target_link_libraries(${PROJECT_NAME}
		libopt
	)
endif()

target_link_libraries(${PROJECT_NAME}
	GPUDistLib
	)

add_dependencies(GPUDistTestSimple GPUDistLib)	

#
# $Log$
#