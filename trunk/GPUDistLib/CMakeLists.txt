# Reference:

#The name of the project
PROJECT(GPUDistLib)

FIND_PACKAGE(CUDA)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(MYLIB_DIR 		"~/mylib" 	CACHE PATH "The path to MyLib.")	

CUDA_INCLUDE_DIRECTORIES(
	${MY_LIB_DIR}/include
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CUDA_SDK_ROOT_DIR}/common/inc
)

link_directories(
	${MY_LIB_DIR}/lib
	${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32
	${CUDA_SDK_ROOT_DIR}/common/lib/Win32
)

# The source codes
CUDA_ADD_LIBRARY(${PROJECT_NAME}
	GPUDistLib.h
	GPUDistLib_host.cu
	kernel_CompDist.h

	# ADD-BY-LEETEN 04/05/2012-BEGIN
	GPUHausdorffDist_host.cu	
	kernel_CompPairDist.h	
	# ADD-BY-LEETEN 04/05/2012-END
#	GPUDistCudpp_host.cu
#	kernel_CompDistCudpp.h
)

# ADD-BY-LEETEN 04/04/2012-BEGIN
set_target_properties(${PROJECT_NAME} PROPERTIES 
	DEBUG_OUTPUT_NAME "${PROJECT_NAME}_d"
	RELEASE_OUTPUT_NAME "${PROJECT_NAME}_r"
	)

# ADD-BY-LEETEN 04/05/2012-BEGIN
target_link_libraries(${PROJECT_NAME}
	cudpp32
)
# ADD-BY-LEETEN 04/05/2012-END
	
install(
	TARGETS ${PROJECT_NAME}
	DESTINATION lib	)	

file(GLOB HEADER_FILES "*.h" )
install(
	FILES ${HEADER_FILES}
	DESTINATION "include/")
# ADD-BY-LEETEN 04/04/2012-END

#
# $Log$
#
