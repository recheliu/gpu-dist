# Reference:

#The name of the project
PROJECT(GPUDistLib)

FIND_PACKAGE(CUDA)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(MYLIB_DIR 		"~/mylib" 	CACHE PATH "The path to MyLib.")	
set(3RDPARTY_DIR 	"~/lib" 	CACHE PATH "The path to 3rd party libraries.")	# ADD-BY-LEETEN 04/16
# ADD-BY-LEETEN 04/07/2012-BEGIN
# DEL-BY-LEETEN 10/04/2012:	set(IS_X64 			OFF 		CACHE BOOL "Is the platform x64?")	
set(WITH_CUDPP 		ON 			CACHE BOOL "Is CUDPP used?")		
# ADD-BY-LEETEN 04/07/2012-END

# ADD-BY-LEETEN 04/09/2012-BEGIN
list(APPEND INC_DIRS
	${MYLIB_DIR}/include
	${3RDPARTY_DIR}/include			# ADD-BY-LEETEN 04/16
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CUDA_SDK_ROOT_DIR}/common/inc
)
# ADD-BY-LEETEN 04/09/2012-END

CUDA_INCLUDE_DIRECTORIES(
	# MOD-BY-LEETEN 04/09/2012-FROM:
		# ${MY_LIB_DIR}/include
		# ${CUDA_TOOLKIT_ROOT_DIR}/include
		# ${CUDA_SDK_ROOT_DIR}/common/inc
	# MOD-BY-LEETEN 04/09/2012-TO:
	${INC_DIRS}
	# MOD-BY-LEETEN 04/09/2012-END
)

# ADD-BY-LEETEN 04/07/2012-BEGIN
INCLUDE_DIRECTORIES(
	# MOD-BY-LEETEN 04/09/2012-FROM:
		# ${MY_LIB_DIR}/include
		# ${CUDA_TOOLKIT_ROOT_DIR}/include
		# ${CUDA_SDK_ROOT_DIR}/common/inc
	# MOD-BY-LEETEN 04/09/2012-TO:
	${INC_DIRS}
	# MOD-BY-LEETEN 04/09/2012-END
)
# ADD-BY-LEETEN 04/07/2012-END

link_directories(
	${3RDPARTY_DIR}/lib			# ADD-BY-LEETEN 04/16
	${MYLIB_DIR}/lib	# MOD-BY-LEETEN 04/09/2012-FROM:	${MY_LIB_DIR}/lib
	${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32
	${CUDA_SDK_ROOT_DIR}/common/lib/Win32
)

# ADD-BY-LEETEN 04/07/2012-BEGIN
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
# ADD-BY-LEETEN 04/07/2012-END

# The source codes
CUDA_ADD_LIBRARY(${PROJECT_NAME}
	GPUDistLib.h
	GPUDistLib_host.cu
	kernel_CompDist.h
	# ADD-BY-LEETEN 04/13/2012-BEGIN
	kernel_CompDistFromTriangle.h	
	kernel_TransformTriangle.h		
	device_Vector.h		
	# ADD-BY-LEETEN 04/13/2012-END
	
	# MOD-BY-LEETEN 04/07/2012-FROM:
		#	# ADD-BY-LEETEN 04/05/2012-BEGIN
		# 	GPUHausdorffDist_host.cu	
		# 	kernel_CompPairDist.h	
		#	# ADD-BY-LEETEN 04/05/2012-END
	
	# 	#	GPUDistCudpp_host.cu
	#	#	kernel_CompDistCudpp.h
	# MOD-BY-LEETEN 04/07/2012-TO:
	GPUDistLib_cpu.cpp
	GPUDistLib_internal.h
	
	${CUDPP_SOURCES}
	# MOD-BY-LEETEN 04/07/2012-END
)

# ADD-BY-LEETEN 04/04/2012-BEGIN
set_target_properties(${PROJECT_NAME} PROPERTIES 
	DEBUG_OUTPUT_NAME "${PROJECT_NAME}_d"
	RELEASE_OUTPUT_NAME "${PROJECT_NAME}_r"
	)

# ADD-BY-LEETEN 04/05/2012-BEGIN
# MOD-BY-LEETEN 04/07/2012-FROM:
	# target_link_libraries(${PROJECT_NAME}
		# cudpp32
	# )
# MOD-BY-LEETEN 04/07/2012-TO:
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
# MOD-BY-LEETEN 04/07/2012-END

# ADD-BY-LEETEN 04/05/2012-END
	
install(
	TARGETS ${PROJECT_NAME}
	DESTINATION lib	)	

file(GLOB HEADER_FILES "*.h" )
install(
	FILES ${HEADER_FILES}
	DESTINATION "include/")
# ADD-BY-LEETEN 04/04/2012-END

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
