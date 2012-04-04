#define _USE_MATH_DEFINES
#include <math.h>

#include <assert.h>	// ADD-BY-LEETEN 03/28/2012

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

#include "kernel_CompDist.h"	// ADD-BY-LEETEN 03/28/2012

static bool bIsUsingCpu;
static bool bIsPrintingTiming;

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
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pfDist_device,
			uNrOfVoxels * sizeof(pfDist_device[0]) ) );
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_BEGIN(bIsPrintingTiming);
	size_t uNrOfBatches = ceilf((float)uNrOfVoxels / (float)BATCH_SIZE);
	for(size_t b = 0; b < uNrOfBatches; b++) 
	{
		// allocate iNrOfElements x iNrOfTimeSteps - 1 threads
		dim3 v3Blk = dim3(BLOCK_DIM_X * BLOCK_DIM_Y);
		size_t uNrOfNeededThreads = (b == uNrOfBatches - 1)?(uNrOfVoxels % BATCH_SIZE):BATCH_SIZE;
		size_t uNrOfBlocks = (unsigned int)ceilf((float)uNrOfNeededThreads / (float)v3Blk.x);
		dim3 v3Grid = dim3(
			// MOD-BY-LEETEN 04/04/2012-FROM:			min(uNrOfBlocks, GRID_DIM_X),
			min(uNrOfBlocks, (size_t)GRID_DIM_X),
			// MOD-BY-LEETEN 04/04/2012-END
			(unsigned int)ceil((double)uNrOfBlocks / (double)GRID_DIM_X)
			);

		// invoke the kernel
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
				&pfDist_device[b * BATCH_SIZE]
			);	
			CUT_CHECK_ERROR("_CompDist_kernel() failed");
		}
	}
LIBCLOCK_END(bIsPrintingTiming);

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

LIBCLOCK_BEGIN(bIsPrintingTiming);
	FREE_MEMORY(pfDist_device);
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_PRINT(bIsPrintingTiming);
}

/*

$Log$

*/