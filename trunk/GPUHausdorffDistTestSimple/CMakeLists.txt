# Reference:

#The name of the project
PROJECT(GPUHausdorffDistTestSimple)

FIND_PACKAGE(CUDA)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(MYLIB_DIR 		"~/mylib" 	CACHE PATH "The path to MyLib.")	
set(3RDPARTY_DIR 	"~/lib" 	CACHE PATH "The path to 3rd party libraries.")	
# ADD-BY-LEETEN 04/07/2012-BEGIN
# DEL-BY-LEETEN 10/04/2012: set(IS_X64 			OFF 		CACHE BOOL "Is the platform x64?")	
set(WITH_CUDPP 		ON 			CACHE BOOL "Is CUDPP used?")		
# ADD-BY-LEETEN 04/07/2012-END

include_directories(
	${MYLIB_DIR}/include	# MOD-BY-LEETEN 04/16/2012:	${MY_LIB_DIR}/include
	${3RDPARTY_DIR}/include	
	${CUDA_TOOLKIT_ROOT_DIR}/include
	${CUDA_SDK_ROOT_DIR}/common/inc
	../GPUDistLib/	
)

link_directories(
	${MYLIB_DIR}/lib	# MOD-BY-LEETEN 04/16/2012:	${MY_LIB_DIR}/lib
	${3RDPARTY_DIR}/lib
	${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32
	${CUDA_SDK_ROOT_DIR}/common/lib/Win32
)

# MOD-BY-LEETEN 04/07/2012-FROM:
	# ADD_EXECUTABLE(${PROJECT_NAME}
		# GPUHausdorffDistTestSimple_main.cpp
	# )
# MOD-BY-LEETEN 04/07/2012-TO:
if( WITH_CUDPP )
	LIST(APPEND SORUCE_FILES
		GPUHausdorffDistTestSimple_main.cpp
	)
else()
	LIST(APPEND SORUCE_FILES
		GPUHausdorffDistNoCudppTest_main.cpp
	)
endif()
ADD_EXECUTABLE(${PROJECT_NAME}
	${SORUCE_FILES}
)
# MOD-BY-LEETEN 04/07/2012-END

# link CUDPP
# if(DEFINED WIN32)	
#	target_link_libraries(${PROJECT_NAME}
#		debug	cudpp32d	optimized	cudpp32
#	)
# else()
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
# endif()
	
# the needed libraries
if(DEFINED WIN32)	
	target_link_libraries(${PROJECT_NAME}
		debug	libopt_d	optimized	libopt_r
		debug	lib3ds-120sd	optimized	lib3ds-120sd	# ADD-BY-LEETEN 04/17/2012
		)
else()
	target_link_libraries(${PROJECT_NAME}
		libopt
	)
endif()

target_link_libraries(${PROJECT_NAME}
	GPUDistLib
	)

# DEL-BY-LEETEN 04/17/2012-BEGIN
# # ADD-BY-LEETEN 04/13/2012-BEGIN
# target_link_libraries(${PROJECT_NAME}
	# lib3ds-120s
# )
# # ADD-BY-LEETEN 04/13/2012-END
# DEL-BY-LEETEN 04/17/2012-END

	
# ADD-BY-LEETEN 04/09/2012-BEGIN
install(
	TARGETS ${PROJECT_NAME}
	RUNTIME	DESTINATION bin
	ARCHIVE	DESTINATION	lib
	)	
# ADD-BY-LEETEN 04/09/2012-END
	
add_dependencies(GPUDistTestSimple GPUDistLib)	

#
# $Log$
#