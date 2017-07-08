# Reference:

#The name of the project
PROJECT(GPUDistLib)

FIND_PACKAGE(CUDA)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(MYLIB_DIR 		"~/mylib" 	CACHE PATH "The path to MyLib.")	
set(3RDPARTY_DIR 	"~/lib" 	CACHE PATH "The path to 3rd party libraries.")	
set(WITH_CUDPP 		ON 			CACHE BOOL "Is CUDPP used?")		

list(APPEND INC_DIRS
	${MYLIB_DIR}/include
	${3RDPARTY_DIR}/include
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CUDA_SDK_ROOT_DIR}/common/inc
)

CUDA_INCLUDE_DIRECTORIES(
	${INC_DIRS}
)

INCLUDE_DIRECTORIES(
	${INC_DIRS}
)

link_directories(
	${3RDPARTY_DIR}/lib			
	${MYLIB_DIR}/lib	
	${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32
	${CUDA_SDK_ROOT_DIR}/common/lib/Win32
)

if( WITH_CUDPP )
	add_definitions(
		-DWITH_CUDPP
	)
	
	LIST(APPEND	CUDPP_SOURCES
		GPUHausdorffDist_host.cu	
		kernel_CompPairDist.h	
		
		GPUDistInBatch_host.cu
		kernel_CompPairDistInBatch.h
		
		kernel_SetupSegBegin.h
	)
endif()

# The source codes
CUDA_ADD_LIBRARY(${PROJECT_NAME}
	GPUDistLib.h
	GPUDistLib_host.cu
	kernel_CompDist.h
	kernel_CompDistFromTriangle.h	
	kernel_TransformTriangle.h		
	device_Vector.h		
	
	GPUDistLib_cpu.cpp
	GPUDistLib_internal.h
	
	${CUDPP_SOURCES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES 
	DEBUG_OUTPUT_NAME "${PROJECT_NAME}_d"
	RELEASE_OUTPUT_NAME "${PROJECT_NAME}_r"
	)

if( WITH_CUDPP )
	if( IS_X64 )
		target_link_libraries(${PROJECT_NAME}
			cudpp64
		)
	else()
		target_link_libraries(${PROJECT_NAME}
			cudpp32
		)
	endif()
endif()
	
install(
	TARGETS ${PROJECT_NAME}
	DESTINATION lib	)	

file(GLOB HEADER_FILES "*.h" )
install(
	FILES ${HEADER_FILES}
	DESTINATION "include/")

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  @ONLY
  )
  
install(
	FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
	DESTINATION ".")

#
# $Log$
#
