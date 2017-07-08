# Reference:

#The name of the project
PROJECT(GPUHausdorffDistTestSimple)

FIND_PACKAGE(CUDA)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(MYLIB_DIR 		"~/mylib" 	CACHE PATH "The path to MyLib.")	
set(3RDPARTY_DIR 	"~/lib" 	CACHE PATH "The path to 3rd party libraries.")	
set(WITH_CUDPP 		ON 			CACHE BOOL "Is CUDPP used?")		

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

# link CUDPP
# if(DEFINED WIN32)	
#	target_link_libraries(${PROJECT_NAME}
#		debug	cudpp32d	optimized	cudpp32
#	)
# else()
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
# endif()

target_link_libraries(${PROJECT_NAME}
	GPUDistLib
	)
	
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