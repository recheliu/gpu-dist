# Reference:
# https://gforge.sci.utah.edu/gf/project/findcuda/scmsvn/?action=browse&path=%2F*checkout*%2Ftrunk%2FFindCuda.html

#The name of the project
PROJECT(GPUDist)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	SET(IS_X64 TRUE)
else()
	SET(IS_X64 FALSE)
endif()

add_subdirectory(GPUDistLib) 	
add_subdirectory(GPUDistTestSimple)
add_subdirectory(GPUHausdorffDistTestSimple)	
add_subdirectory(GPUDistToTrianglesSimple)		
if( WITH_CUDPP )	
	add_subdirectory(GPUDistInBatchTestSimple)		
endif()				

#
# $Log$
#
