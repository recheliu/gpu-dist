#if	0	// MOD-BY-LEETEN 04/07/2012-FROM:
	#define _USE_MATH_DEFINES
	#include <math.h>

	#include <assert.h>	// ADD-BY-LEETEN 03/28/2012
	#include <cudpp.h>	// ADD-BY-LEETEN 04/05/2012

	#ifdef	 WIN32
		#undef	 WIN32
		#include "liblog.h"
		#define  WIN32
	#else
		#include "liblog.h"
	#endif
	#include "cuda_macro.h"	
	#include "cuda_macro3d.h"	
	#include "libclock.h"

	#include "GPUDistLib.h"	

	#define BLOCK_DIM_X	16
	#define BLOCK_DIM_Y	8	
	#define GRID_DIM_X	1024
	#define GRID_DIM_Y	1024
	#define BATCH_SIZE	(BLOCK_DIM_X * BLOCK_DIM_Y * GRID_DIM_X * GRID_DIM_Y)
#else	// MOD-BY-LEETEN 04/07/2012-TO:
	#include "GPUDistLib_internal.h"
#endif	// MOD-BY-LEETEN 04/07/2012-END

#if	0	// DEL-BY-LEETEN 04/04/2012-BEGIN
	__constant__ int	iNrOfBasis_const;
	__constant__ int	iNrOfTimeSteps_const;

	int iNrOfBasis;
	int iNrOfTimeSteps;

	//! A 2D texture of the time difference matrices
	cudaPitchedPtr cTimeDiffMatrices_pitched;
	texture<float, 2, cudaReadModeElementType> t2DfTimeDiffMatrices;

	// ADD-BY-LEETEN 03/28/2012-BEGIN
	#define PRINT_PomCompComplexity_TIMING	1
	texture<float, 2, cudaReadModeElementType> t2DfMatrix;
	// ADD-BY-LEETEN 03/28/2012-END
	texture<float, 2, cudaReadModeElementType> t2DfCoefs;
#endif		// DEL-BY-LEETEN 04/04/2012-END

// ADD-BY-LEETEN 04/07/2012-BEGIN
#if	IS_COMP_MULTI_POINTS
__constant__ float4	pf4Points_const[MAX_NR_OF_COMP_POINTS];
#endif	// #if	IS_COMP_MULTI_POINTS
// ADD-BY-LEETEN 04/07/2012-END

#include "kernel_CompDist.h"	// ADD-BY-LEETEN 03/28/2012

#if	0	// MOD-BY-LEETEN 04/05/2012-FROM:
	static bool bIsUsingCpu;
	static bool bIsPrintingTiming;
#else		// MOD-BY-LEETEN 04/05/2012-TO:
bool bIsUsingCpu;
bool bIsPrintingTiming;
#endif		// MOD-BY-LEETEN 04/05/2012-END

// ADD-BY-LEETEN 04/05/2012-BEGIN
__constant__ int	iDummy_const;

#if	defined(WITH_CUDPP)
CUDPPHandle cudpp;
#endif	// #if	defined(WITH_CUDPP)

void
_GPUDistFree()
{
#if	defined(WITH_CUDPP)
	ASSERT_OR_LOG(
		CUDPP_SUCCESS == cudppDestroy(cudpp),
		"");
#endif	// #if	defined(WITH_CUDPP)
}

void
_GPUDistInit()
{
	int iDummy = 0;
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMemcpyToSymbol("iDummy_const",	&iDummy,	sizeof(iDummy),	0, cudaMemcpyHostToDevice) );

#if	defined(WITH_CUDPP)
	ASSERT_OR_LOG(
		CUDPP_SUCCESS == cudppCreate(&cudpp),
		"");
#endif	// #if	defined(WITH_CUDPP)
	atexit(_GPUDistFree);
}
// ADD-BY-LEETEN 04/05/2012-END

void
_GPUDistUseCpu
(
		bool bIsUsingCpu
		)
{
	::bIsUsingCpu = bIsUsingCpu;
}

void
_GPUDistPrintTiming
(
		bool bIsPrintingTiming
		)
{
	::bIsPrintingTiming = bIsPrintingTiming;
}

#if	0	// DEL-BY-LEETEN 04/07/2012-BEGIN
	void
	_GPUDistComputeDistanceFieldFromPointsByCpu
	(
		size_t uNrOfPoints,
		float4 pf4Points[],
		size_t uWidth,
		size_t uHeight,
		size_t uDepth,
		float pfDist[]
	)
	{
	LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);
	LIBCLOCK_BEGIN(bIsPrintingTiming);
		for(size_t v = 0,	d = 0; d < uDepth; d++)
			for(size_t		h = 0; h < uHeight; h++)
				for(size_t	w = 0; w < uWidth; w++, v++)
				{
					float fDist = (float)HUGE_VAL;
					for(size_t p = 0; p < uNrOfPoints; p++)
					{
						float fDx = (float)w - pf4Points[p].x;
						float fDy = (float)h - pf4Points[p].y;
						float fDz = (float)d - pf4Points[p].z;
						float fD = sqrt(fDx * fDx + fDy * fDy + fDz * fDz);
						fDist = min(fDist, fD);
					}
					pfDist[v] = sqrtf(fDist);
				}
	LIBCLOCK_END(bIsPrintingTiming);
	LIBCLOCK_PRINT(bIsPrintingTiming);
	}
#endif	// DEL-BY-LEETEN 04/07/2012-END

void
_GPUDistComputeDistanceFieldFromPoints
(
	size_t uNrOfPoints,
	float4 pf4Points[],
	size_t uWidth,
	size_t uHeight,
	size_t uDepth,
	float pfDist[]
)
{
	if( bIsUsingCpu )
	{
		_GPUDistComputeDistanceFieldFromPointsByCpu
			(
				uNrOfPoints,
				pf4Points,
				uWidth,
				uHeight,
				uDepth,
				pfDist
			);

		return;
	}
LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);
LIBCLOCK_BEGIN(bIsPrintingTiming);
	size_t uNrOfVoxels = uWidth * uHeight * uDepth;

	// allocate a 2D linear buffer for the time difference
	float *pfDist_device;
	#if	0	// MOD-BY-LEETEN 04/07/2012-FROM:
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMalloc(
				&pfDist_device,
				uNrOfVoxels * sizeof(pfDist_device[0]) ) );
	#else		// MOD-BY-LEETEN 04/07/2012-TO:
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pfDist_device,
			BATCH_SIZE * sizeof(pfDist_device[0]) ) );

	float *pfDist_host;
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMallocHost(
			&pfDist_host,
			BATCH_SIZE * sizeof(pfDist_host[0]) ) );
	#endif		// MOD-BY-LEETEN 04/07/2012-END
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_BEGIN(bIsPrintingTiming);
	size_t uNrOfBatches = ceilf((float)uNrOfVoxels / (float)BATCH_SIZE);
	// ADD-BY-LEETEN 04/07/2012-BEGIN
	size_t uNrOfThreadsLastBatch = uNrOfVoxels % BATCH_SIZE;
	uNrOfThreadsLastBatch = (!uNrOfThreadsLastBatch)?BATCH_SIZE:uNrOfThreadsLastBatch;
	// ADD-BY-LEETEN 04/07/2012-END
	for(size_t b = 0; b < uNrOfBatches; b++) 
	{
		// allocate iNrOfElements x iNrOfTimeSteps - 1 threads
		dim3 v3Blk = dim3(BLOCK_DIM_X * BLOCK_DIM_Y);
		// MOD-BY-LEETEN 04/07/2012-FROM:		size_t uNrOfNeededThreads = (b == uNrOfBatches - 1)?(uNrOfVoxels % BATCH_SIZE):BATCH_SIZE;
		size_t uNrOfNeededThreads = (b == uNrOfBatches - 1)?uNrOfThreadsLastBatch:BATCH_SIZE;
		// MOD-BY-LEETEN 04/07/2012-END
		size_t uNrOfBlocks = (unsigned int)ceilf((float)uNrOfNeededThreads / (float)v3Blk.x);
		dim3 v3Grid = dim3(
			// MOD-BY-LEETEN 04/04/2012-FROM:			min(uNrOfBlocks, GRID_DIM_X),
			min(uNrOfBlocks, (size_t)GRID_DIM_X),
			// MOD-BY-LEETEN 04/04/2012-END
			(unsigned int)ceil((double)uNrOfBlocks / (double)GRID_DIM_X)
			);

		// invoke the kernel

		// ADD-BY-LEETEN 04/07/2012-BEGIN
		#if	IS_COMP_MULTI_POINTS	
		for(size_t p = 0; p < uNrOfPoints; p+=MAX_NR_OF_COMP_POINTS)
		{
			size_t uNrOfPointsToCompare = min(uNrOfPoints - p, (size_t)MAX_NR_OF_COMP_POINTS);
			CUDA_SAFE_CALL_NO_SYNC( 
				cudaMemcpyToSymbol(
					"pf4Points_const",	
					&pf4Points[p],	
					sizeof(pf4Points[0]) * uNrOfPointsToCompare,
					0, cudaMemcpyHostToDevice) );

			_CompDist_kernel<<<v3Grid, v3Blk, 0>>>
			(
				b * BATCH_SIZE,
				p,
				uNrOfPointsToCompare,
				uWidth,
				uHeight,
				uDepth,
				&pfDist_device[0]
			);	
			CUT_CHECK_ERROR("_CompDist_kernel() failed");
		}
		#else	// #if	IS_COMP_MULTI_POINTS	
		// ADD-BY-LEETEN 04/07/2012-END
		for(size_t p = 0; p < uNrOfPoints; p++)
		{
			_CompDist_kernel<<<v3Grid, v3Blk, 0>>>
			(
				b * BATCH_SIZE,
				p,
				pf4Points[p],
				uWidth,
				uHeight,
				uDepth,
				// MOD-BY-LEETEN 04/07/2012-FROM:				&pfDist_device[b * BATCH_SIZE]
				&pfDist_device[0]
				// MOD-BY-LEETEN 04/07/2012-END
			);	
			CUT_CHECK_ERROR("_CompDist_kernel() failed");
		}
		#endif	// #if	IS_COMP_MULTI_POINTS	// ADD-BY-LEETEN 04/07/2012

		// ADD-BY-LEETEN 04/07/2012-BEGIN
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				pfDist_host, 
				pfDist_device,
				uNrOfNeededThreads * sizeof(pfDist_host[0]),
				cudaMemcpyDeviceToHost) );
		memcpy(&pfDist[b * BATCH_SIZE], &pfDist_host[0], uNrOfNeededThreads * sizeof(pfDist[0]));
		// ADD-BY-LEETEN 04/07/2012-END
	}
LIBCLOCK_END(bIsPrintingTiming);

#if	0	// DEL-BY-LEETEN 04/07/2012-BEGIN
	LIBCLOCK_BEGIN(bIsPrintingTiming);
		// download the result back to the host
		#if	0	// MOD-BY-LEETEN 04/04/2012-FROM:
			CUDA_SAFE_CALL_NO_SYNC( 
				cudaMemcpy(
					pfDist, 
					pfDist_device,
					uNrOfVoxels * sizeof(pfDist),
					cudaMemcpyDeviceToHost) );
		#else		// MOD-BY-LEETEN 04/04/2012-TO:
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				pfDist, 
				pfDist_device,
				uNrOfVoxels * sizeof(pfDist[0]),
				cudaMemcpyDeviceToHost) );
		#endif		// MOD-BY-LEETEN 04/04/2012-END

	LIBCLOCK_END(bIsPrintingTiming);
#endif	// DEL-BY-LEETEN 04/07/2012-END

LIBCLOCK_BEGIN(bIsPrintingTiming);
	FREE_MEMORY_ON_HOST(pfDist_host);
	FREE_MEMORY(pfDist_device);
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_PRINT(bIsPrintingTiming);
}

// ADD-BY-LEETEN 04/07/2012-BEGIN
void
_GPUDistCompDistFromPointsToPoints
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	float pfDist[]
)
{
	if( bIsUsingCpu )
	{
		_GPUDistCompDistFromPointsToPointsByCpu
		(
			uNrOfPoints1,
			pf4Points1,

			uNrOfPoints2,
			pf4Points2,

			pfDist
		);
		return;
	}

LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);
LIBCLOCK_BEGIN(bIsPrintingTiming);
	// allocate a linear buffer for the time difference
	float4 *pf4Points1_device;
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pf4Points1_device,
			BATCH_SIZE * sizeof(pf4Points1_device[0]) ) );

	float *pfDist_device;
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pfDist_device,
			BATCH_SIZE * sizeof(pfDist_device[0]) ) );

	float *pfDist_host;
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMallocHost(
			&pfDist_host,
			BATCH_SIZE * sizeof(pfDist_host[0]) ) );
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_BEGIN(bIsPrintingTiming);
	size_t uNrOfBatches = ceilf((float)uNrOfPoints1 / (float)BATCH_SIZE);
	size_t uNrOfThreadsLastBatch = uNrOfPoints1 % BATCH_SIZE;
	uNrOfThreadsLastBatch = (!uNrOfThreadsLastBatch)?BATCH_SIZE:uNrOfThreadsLastBatch;
	for(size_t b = 0; b < uNrOfBatches; b++) 
	{
		size_t uNrOfNeededThreads = (b == uNrOfBatches - 1)?uNrOfThreadsLastBatch:BATCH_SIZE;
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				&pf4Points1_device[0], 
				&pf4Points1[b * BATCH_SIZE],
				uNrOfNeededThreads * sizeof(pf4Points1_device[0]),
				cudaMemcpyHostToDevice) );

		// allocate iNrOfElements x iNrOfTimeSteps - 1 threads
		dim3 v3Blk = dim3(BLOCK_DIM_X * BLOCK_DIM_Y);
		size_t uNrOfBlocks = (unsigned int)ceilf((float)uNrOfNeededThreads / (float)v3Blk.x);
		dim3 v3Grid = dim3(
			min(uNrOfBlocks, (size_t)GRID_DIM_X),
			(unsigned int)ceil((double)uNrOfBlocks / (double)GRID_DIM_X)
			);

		// invoke the kernel

		// ADD-BY-LEETEN 04/07/2012-BEGIN
		#if	IS_COMP_MULTI_POINTS	
		for(size_t p2 = 0; p2 < uNrOfPoints2; p2+=MAX_NR_OF_COMP_POINTS)
		{
			size_t uNrOfPoints2ToCompare = min(uNrOfPoints2 - p2, (size_t)MAX_NR_OF_COMP_POINTS);
			CUDA_SAFE_CALL_NO_SYNC( 
				cudaMemcpyToSymbol(
					"pf4Points_const",	
					&pf4Points2[p2],	
					sizeof(pf4Points2[0]) * uNrOfPoints2ToCompare,
					0, cudaMemcpyHostToDevice) );

			_CompDistFromPoints_kernel<<<v3Grid, v3Blk, 0>>>
			(
				uNrOfNeededThreads,
				&pf4Points1_device[0],

				p2,
				uNrOfPoints2ToCompare,

				&pfDist_device[0]
			);	
			CUT_CHECK_ERROR("_CompDist_kernel() failed");
		}
		#else	// #if	IS_COMP_MULTI_POINTS	
		// ADD-BY-LEETEN 04/07/2012-END
		for(size_t p2 = 0; p2 < uNrOfPoints2; p2++)
		{
			_CompDistFromPoints_kernel<<<v3Grid, v3Blk, 0>>>
			(
				uNrOfNeededThreads,
				&pf4Points1_device[0],

				p2,
				pf4Points2[p2],

				&pfDist_device[0]
			);	
			CUT_CHECK_ERROR("_CompDist_kernel() failed");
		}
		#endif	// #if	IS_COMP_MULTI_POINTS	// ADD-BY-LEETEN 04/07/2012
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				pfDist_host, 
				pfDist_device,
				uNrOfNeededThreads * sizeof(pfDist_host[0]),
				cudaMemcpyDeviceToHost) );
		memcpy(&pfDist[b * BATCH_SIZE], &pfDist_host[0], uNrOfNeededThreads * sizeof(pfDist[0]));
	}
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_BEGIN(bIsPrintingTiming);
	FREE_MEMORY_ON_HOST(pfDist_host);
	FREE_MEMORY(pfDist_device);
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_PRINT(bIsPrintingTiming);
}
// ADD-BY-LEETEN 04/07/2012-END

/*

$Log$

*/