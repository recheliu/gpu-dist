# Reference:
# https://gforge.sci.utah.edu/gf/project/findcuda/scmsvn/?action=browse&path=%2F*checkout*%2Ftrunk%2FFindCuda.html

#The name of the project
PROJECT(GPUDist)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

add_subdirectory(GPUDistLib) 	
add_subdirectory(GPUDistTestSimple)
add_subdirectory(GPUHausdorffDistTestSimple)	# ADD-BY-LEETEN 04/05/2012
# ADD-BY-LEETEN 04/07/2012-BEGIN
if( WITH_CUDPP )	
	add_subdirectory(GPUDistInBatchTestSimple)		
endif()				
# ADD-BY-LEETEN 04/07/2012-END

#
# $Log$
#
