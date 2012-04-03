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

#	GPUDistCudpp_host.cu
#	kernel_CompDistCudpp.h
)

#
# $Log: CMakeLists.txt,v $
# Revision 1.2  2012-03-30 06:00:33  leeten
#
# [03/29/2012]
# <POMCuda_host.cu>
# <POMCuda.h>
# 1. [ADD] Include the header assert.h.
# 2. [ADD] Define functions _PomCompComplexityFree(), _PomCompComplexityInit(), and _PomCompComplexity() to compute the complexity of all POMs on GPUs.
#
# <CMakeLists.txt>
# <kernel_CompComplexity.h>
# 1. [1ST] Check in the kernel to compute the complexity.
#
# Revision 1.1  2012-03-19 18:52:55  leeten
#
# [03/19/2012]
# <CMakeLists.txt>
# 1. [1ST] Check in the Cmake files, the kernel, the host files, and the C/C+ header for GPU-based computation of POM related metrics.
#
#
