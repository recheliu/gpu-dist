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

// DEL-BY-LEETEN 04/15/2012:	#include <lib3ds/vector.h>

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
// ADD-BY-LEETEN 04/13/2012-BEGIN
#include "kernel_CompDistFromTriangle.h"	
#include "kernel_TransformTriangle.h"	
// ADD-BY-LEETEN 04/13/2012-END

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

// ADD-BY-LEETEN 04/13/2012-BEGIN
bool
BGPUDistIsDistSquaredRoot
(
		)
{
	#if	IS_SQRT
	return true;
	#else	// #if	IS_SQRT
	return false;
	#endif	// #if	IS_SQRT
}
// ADD-BY-LEETEN 04/13/2012-END

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

// ADD-BY-LEETEN 04/13/2012-BEGIN
void
_GPUDistCompDistFromPointsToTriangles
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	size_t uNrOfTriangles,
	ulong4 pu4TriangleVertices[],

	bool bIsPrecomputingTrasforms,

	float pfDists[]
)
{
	if( bIsUsingCpu )
	{
		_GPUDistCompDistFromPointsToTrianglesByCpu
		(
			uNrOfPoints1,
			pf4Points1,

			uNrOfPoints2,
			pf4Points2,

			uNrOfTriangles,
			pu4TriangleVertices,

			bIsPrecomputingTrasforms,

			pfDists
		);
		return;
	}

LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);

// DEL-BY-LEETEN 04/15/2012:	LIBCLOCK_BEGIN(bIsPrintingTiming);
	TBuffer<float4> pf4Xs;
	TBuffer<float4> pf4Ys;
	TBuffer<float4> pf4Zs;
	TBuffer<float4> pf4B2s;
	TBuffer<float4> pf4C2s;
	TBuffer<float>	pfDets;
	if(bIsPrecomputingTrasforms)
	{
		LIBCLOCK_BEGIN(bIsPrintingTiming);	// ADD-BY-LEETEN 04/15/2012
		pf4Xs.alloc(uNrOfTriangles);
		pf4Ys.alloc(uNrOfTriangles);
		pf4Zs.alloc(uNrOfTriangles);
		pf4B2s.alloc(uNrOfTriangles);
		pf4C2s.alloc(uNrOfTriangles);
		pfDets.alloc(uNrOfTriangles);

		ulong4 *pu4TriangleVertices_device;	
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMalloc(
				&pu4TriangleVertices_device, 
				uNrOfTriangles * sizeof(pu4TriangleVertices_device[0]) ) );
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				pu4TriangleVertices_device,	
				pu4TriangleVertices, 
				uNrOfTriangles * sizeof(pu4TriangleVertices_device[0]), cudaMemcpyHostToDevice) );

		// compute the transform for all triangles
		float4 *pf4Xs_device;	CUDA_SAFE_CALL_NO_SYNC( cudaMalloc(&pf4Xs_device, uNrOfTriangles * sizeof(pf4Xs_device[0]) ) );
		float4 *pf4Ys_device;	CUDA_SAFE_CALL_NO_SYNC( cudaMalloc(&pf4Ys_device, uNrOfTriangles * sizeof(pf4Ys_device[0]) ) );
		float4 *pf4Zs_device;	CUDA_SAFE_CALL_NO_SYNC( cudaMalloc(&pf4Zs_device, uNrOfTriangles * sizeof(pf4Zs_device[0]) ) );
		float4 *pf4B2s_device;	CUDA_SAFE_CALL_NO_SYNC( cudaMalloc(&pf4B2s_device, uNrOfTriangles * sizeof(pf4B2s_device[0]) ) );
		float4 *pf4C2s_device;	CUDA_SAFE_CALL_NO_SYNC( cudaMalloc(&pf4C2s_device, uNrOfTriangles * sizeof(pf4C2s_device[0]) ) );
		float *pfDets_device;	CUDA_SAFE_CALL_NO_SYNC( cudaMalloc(&pfDets_device, uNrOfTriangles * sizeof(pfDets_device[0]) ) );

		////////// upload all points2 as a 2D texture to the device //////////////
		float4 *pf4Points_device;	
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMalloc(
				&pf4Points_device,
				uNrOfPoints2 * sizeof(pf4Points_device[0]) ) );
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				pf4Points_device,	
				pf4Points2, 
				uNrOfPoints2 * sizeof(pf4Points_device[0]), cudaMemcpyHostToDevice) );
		size_t uPointTexWidth = 4096;
		size_t uPointTexHeight = (size_t)ceil((double)uNrOfPoints2 / (double)uPointTexWidth);

		/*
		SETUP_TEXTURE(
			t2Df4Points, 
			cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp, 
			cudaFilterModePoint, false);
		*/
		t2Df4Points.normalized = false;
		t2Df4Points.filterMode = cudaFilterModePoint;
		for(int a = 0; a < 3; a ++)
			t2Df4Points.addressMode[a] = cudaAddressModeClamp;
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaBindTexture2D(
				NULL, 
				t2Df4Points, 
				&pf4Points_device[0],
				t2Df4Points.channelDesc,
				uPointTexWidth,
				uPointTexHeight,
				uPointTexWidth * sizeof(pf4Points_device[0])));

		// allocate iNrOfElements x iNrOfTimeSteps - 1 threads
		dim3 v3Blk = dim3(BLOCK_DIM_X * BLOCK_DIM_Y);
		size_t uNrOfBlocks = (unsigned int)ceilf((float)uNrOfTriangles / (float)v3Blk.x);
		dim3 v3Grid = dim3(
			min(uNrOfBlocks, (size_t)GRID_DIM_X),
			(unsigned int)ceil((double)uNrOfBlocks / (double)GRID_DIM_X)
			);

		_TransformTriangle_kernel<<<v3Grid, v3Blk, 0>>>
		(
			uNrOfTriangles,
			uPointTexWidth,
			uPointTexHeight,

			pu4TriangleVertices_device,

			pf4Xs_device,
			pf4Ys_device,
			pf4Zs_device,
			pf4B2s_device,
			pf4C2s_device,
			pfDets_device);
		CUT_CHECK_ERROR("_TransformTriangle_kernel() failed");

		CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(&pf4Xs[0],	pf4Xs_device, uNrOfTriangles * sizeof(pf4Xs_device[0]), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(&pf4Ys[0],	pf4Ys_device, uNrOfTriangles * sizeof(pf4Ys_device[0]), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(&pf4Zs[0],	pf4Zs_device, uNrOfTriangles * sizeof(pf4Zs_device[0]), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(&pf4B2s[0],	pf4B2s_device, uNrOfTriangles * sizeof(pf4B2s_device[0]), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(&pf4C2s[0],	pf4C2s_device, uNrOfTriangles * sizeof(pf4C2s_device[0]), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL_NO_SYNC( cudaMemcpy(&pfDets[0],	pfDets_device, uNrOfTriangles * sizeof(pfDets_device[0]), cudaMemcpyDeviceToHost ) );

		FREE_MEMORY(pf4Xs_device);
		FREE_MEMORY(pf4Ys_device);
		FREE_MEMORY(pf4Zs_device);
		FREE_MEMORY(pf4B2s_device);
		FREE_MEMORY(pf4C2s_device);
		FREE_MEMORY(pfDets_device);
		FREE_MEMORY(pu4TriangleVertices_device);
		LIBCLOCK_END(bIsPrintingTiming);	// ADD-BY-LEETEN 04/15/2012
	}
// 
// DEL-BY-LEETEN 04/15/2012:	LIBCLOCK_END(bIsPrintingTiming);

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
		for(size_t t = 0; t < uNrOfTriangles; t++)
		{
			///////////////////////////// compute ///////////////////////////////
			// use A as the origin
			// use BA as the Z axis
			float4 f4A = pf4Points2[pu4TriangleVertices[t].x];

			if( !bIsPrecomputingTrasforms )
			{
				float4 f4B = pf4Points2[pu4TriangleVertices[t].y];
				float4 f4C = pf4Points2[pu4TriangleVertices[t].z];

				#if	0	// MOD-BY-LEETEN 04/15/2012-FROM:
					Lib3dsVector v3BA;
					v3BA[0] = f4B.x - f4A.x;
					v3BA[1] = f4B.y - f4A.y;
					v3BA[2] = f4B.z - f4A.z;
					Lib3dsVector v3Z;
					lib3ds_vector_copy(v3Z, v3BA);
					lib3ds_vector_normalize(v3Z);
					/*
					float fZ = lib3ds_vector_length(v3Z);
					if( fZ > 0.0f )
						for(int c = 0; c < 3; c++)
							v3Z[c] /= fZ;
					*/



					// use the normal N of ABC (ie. the cross product of CA and BA) as X axis
					Lib3dsVector v3CA;
					v3CA[0] = f4C.x - f4A.x;
					v3CA[1] = f4C.y - f4A.y;
					v3CA[2] = f4C.z - f4A.z;
					Lib3dsVector v3X;
					lib3ds_vector_cross(v3X, v3CA, v3BA);
					lib3ds_vector_normalize(v3X);
					/*
					float fX = lib3ds_vector_length(v3X);
					if( fX > 0.0f )
						for(int c = 0; c < 3; c++)
							v3X[c] /= fX;
					*/

					// use the cross product of BA and N as the y axis
					Lib3dsVector v3Y;
					lib3ds_vector_cross(v3Y, v3BA, v3X);
					lib3ds_vector_normalize(v3Y);
					/*
					float fY = lib3ds_vector_length(v3Y);
					if( fY > 0.0f )
						for(int c = 0; c < 3; c++)
							v3Y[c] /= fY;
					*/

					
					// transform B and C to the new coordinate
					Lib3dsVector v3B2;
					v3B2[0] = lib3ds_vector_dot(v3BA, v3X);
					v3B2[1] = lib3ds_vector_dot(v3BA, v3Y);
					v3B2[2] = lib3ds_vector_dot(v3BA, v3Z);

					Lib3dsVector v3C2;
					v3C2[0] = lib3ds_vector_dot(v3CA, v3X);
					v3C2[1] = lib3ds_vector_dot(v3CA, v3Y);
					v3C2[2] = lib3ds_vector_dot(v3CA, v3Z);

					float fDet = v3B2[1] * v3C2[2] - v3B2[2] * v3C2[1];
				#else	// MOD-BY-LEETEN 04/15/2012-TO:
				Lib3dsVector v3X, v3Y, v3Z, v3B2, v3C2;
				float fDet;
				_CompTransform
				(
					f4A, f4B, f4C,
					v3X, v3Y, v3Z, 
					v3B2, v3C2, fDet
				 );
				#endif	// MOD-BY-LEETEN 04/15/2012-END

				CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4A_const",	&f4A,	sizeof(f4A),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4X_const",	&v3X,	3 * sizeof(v3X[0]),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4Y_const",	&v3Y,	3 * sizeof(v3Y[0]),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4Z_const",	&v3Z,	3 * sizeof(v3Z[0]),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4B2_const",&v3B2,	3 * sizeof(v3B2[0]),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4C2_const",&v3C2,	3 * sizeof(v3C2[0]),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("fDet_const",&fDet,	sizeof(fDet),	0, cudaMemcpyHostToDevice) );
			}
			else
			{
				CUDA_SAFE_CALL_NO_SYNC( 
					cudaMemcpyToSymbol("f4A_const",	&f4A,		sizeof(f4A),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( 
					cudaMemcpyToSymbol("f4X_const",	&pf4Xs[t],	sizeof(pf4Xs[0]),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( 
					cudaMemcpyToSymbol("f4Y_const",	&pf4Ys[t],	sizeof(pf4Ys[0]),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( 
					cudaMemcpyToSymbol("f4Z_const",	&pf4Zs[t],	sizeof(pf4Zs[0]),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( 
					cudaMemcpyToSymbol("f4B2_const",&pf4B2s[t],	sizeof(pf4B2s[0]),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( 
					cudaMemcpyToSymbol("f4C2_const",&pf4C2s[t],	sizeof(pf4C2s[0]),	0, cudaMemcpyHostToDevice) );
				CUDA_SAFE_CALL_NO_SYNC( 
					cudaMemcpyToSymbol("fDet_const",&pfDets[t],	sizeof(pfDets[0]),	0, cudaMemcpyHostToDevice) );
			}

			///////////////////////////// invoke the kernel //////////////////////////
			_CompDistFromTriangle_kernel<<<v3Grid, v3Blk, 0>>>
			(
				uNrOfNeededThreads,
				&pf4Points1_device[0],
				t,
				&pfDist_device[0]
			);	
			CUT_CHECK_ERROR("_CompDistFromTriangle_kernel() failed");
		}
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				pfDist_host, 
				pfDist_device,
				uNrOfNeededThreads * sizeof(pfDist_host[0]),
				cudaMemcpyDeviceToHost) );
		memcpy(&pfDists[b * BATCH_SIZE], &pfDist_host[0], uNrOfNeededThreads * sizeof(pfDists[0]));
	}
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_BEGIN(bIsPrintingTiming);
	FREE_MEMORY_ON_HOST(pfDist_host);
	FREE_MEMORY(pfDist_device);
	FREE_MEMORY(pf4Points1_device);	// ADD-BY-LEETEN 04/15/2012
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_PRINT(bIsPrintingTiming);
}
// ADD-BY-LEETEN 04/13/2012-END

// ADD-BY-LEETEN 04/15/2012-BEGIN
void
_GPUDistCountIntersectingTriangles
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	size_t uNrOfTriangles,
	ulong4 pu4TriangleVertices[],

	float4 f4Dir,

	float pfCount[]
)
{
LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);

LIBCLOCK_BEGIN(bIsPrintingTiming);
	float4 *pf4Points1_device;
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pf4Points1_device,
			BATCH_SIZE * sizeof(pf4Points1_device[0]) ) );

	float *pfCount_device;
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pfCount_device,
			BATCH_SIZE * sizeof(pfCount_device[0]) ) );
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMemset(
			pfCount_device,
			0,
			BATCH_SIZE * sizeof(pfCount_device[0]) ) );

	float *pfCount_host;
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMallocHost(
			&pfCount_host,
			BATCH_SIZE * sizeof(pfCount_host[0]) ) );


	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMemcpyToSymbol("f4Dir_const",	&f4Dir,		sizeof(f4Dir),	0, cudaMemcpyHostToDevice) );

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
		for(size_t t = 0; t < uNrOfTriangles; t++)
		{
			///////////////////////////// compute ///////////////////////////////
			// use A as the origin
			// use BA as the Z axis
			float4 f4A = pf4Points2[pu4TriangleVertices[t].x];

			float4 f4B = pf4Points2[pu4TriangleVertices[t].y];
			float4 f4C = pf4Points2[pu4TriangleVertices[t].z];

			Lib3dsVector v3X, v3Y, v3Z, v3B2, v3C2;
			float fDet;
			_CompTransform
			(
				f4A, f4B, f4C,
				v3X, v3Y, v3Z, 
				v3B2, v3C2, fDet
			 );

			CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4A_const",	&f4A,	sizeof(f4A),	0, cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4X_const",	&v3X,	3 * sizeof(v3X[0]),	0, cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4Y_const",	&v3Y,	3 * sizeof(v3Y[0]),	0, cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4Z_const",	&v3Z,	3 * sizeof(v3Z[0]),	0, cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4B2_const",&v3B2,	3 * sizeof(v3B2[0]),	0, cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("f4C2_const",&v3C2,	3 * sizeof(v3C2[0]),	0, cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL_NO_SYNC( cudaMemcpyToSymbol("fDet_const",&fDet,	sizeof(fDet),	0, cudaMemcpyHostToDevice) );

			///////////////////////////// invoke the kernel //////////////////////////
			_CountIntersectingTriangle_kernel<<<v3Grid, v3Blk, 0>>>
			(
				uNrOfNeededThreads,
				&pf4Points1_device[0],
				&pfCount_device[0]
			);	
			CUT_CHECK_ERROR("_CompDistFromTriangle_kernel() failed");
		}
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				pfCount_host, 
				pfCount_device,
				uNrOfNeededThreads * sizeof(pfCount_host[0]),
				cudaMemcpyDeviceToHost) );
		memcpy(&pfCount[b * BATCH_SIZE], &pfCount_host[0], uNrOfNeededThreads * sizeof(pfCount[0]));
	}
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_BEGIN(bIsPrintingTiming);
	FREE_MEMORY_ON_HOST(	pfCount_host		);
	FREE_MEMORY(		pfCount_device		);
	FREE_MEMORY(		pf4Points1_device	);
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_PRINT(bIsPrintingTiming);
}
// ADD-BY-LEETEN 04/15/2012-END


/*

$Log$

*/