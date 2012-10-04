# Reference:
# https://gforge.sci.utah.edu/gf/project/findcuda/scmsvn/?action=browse&path=%2F*checkout*%2Ftrunk%2FFindCuda.html

#The name of the project
PROJECT(GPUDist)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# ADD-BY-LEETEN 10/04/2012-BEGIN
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	SET(IS_X64 TRUE)
else()
	SET(IS_X64 FALSE)
endif()
# ADD-BY-LEETEN 10/04/2012-END

add_subdirectory(GPUDistLib) 	
add_subdirectory(GPUDistTestSimple)
add_subdirectory(GPUHausdorffDistTestSimple)	# ADD-BY-LEETEN 04/05/2012
add_subdirectory(GPUDistToTrianglesSimple)		# ADD-BY-LEETEN 04/13/2012
# ADD-BY-LEETEN 04/07/2012-BEGIN
if( WITH_CUDPP )	
	add_subdirectory(GPUDistInBatchTestSimple)		
endif()				
# ADD-BY-LEETEN 04/07/2012-END

#
# $Log$
#
