#define _USE_MATH_DEFINES
#include <math.h>

#include <assert.h>	// ADD-BY-LEETEN 03/28/2012

#include "cudpp.h"

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
#include "libbuf.h"	

#include "GPUDistLib.h"	

#define BLOCK_DIM_X	16
#define BLOCK_DIM_Y	8
#define GRID_DIM_X	1024
#define GRID_DIM_Y	1024
#define BATCH_SIZE	(BLOCK_DIM_X * BLOCK_DIM_Y * GRID_DIM_X * GRID_DIM_Y)

#define	IS_COMPACT_AND_REDUCE_ON_CPU	0
#if		!IS_COMPACT_AND_REDUCE_ON_CPU	
	#define	IS_REDUCE_ON_CPU		0
#endif	// #if	!IS_COMPACT_AND_REDUCE_ON_CPU	

//! Decide how the segments are initialized
/*!
IS_MEMCPY_SEG = 1: Initialize the value on the host and then copy to the device
IS_MEMCPY_SEG = 0: Initialize the value on the device side by using the kernel _SetupSegments_kernel
*/
#define	IS_MEMCPY_SEG			0

struct CConfigPlan{
	CUDPPConfiguration cConfig;
	CUDPPHandle hPlan;
	CConfigPlan()
	{
		hPlan = 0;
	}
	~CConfigPlan()
	{
		if( hPlan )
			cudppDestroyPlan(hPlan);
	}
} 
	cFindMinDistFromPoint1ToPoints2, 
	cCompactMinDistFromPoint1ToPoints2,
	cReduceMinDistFromPoint1ToPoints2;

#include "kernel_CompPairDist.h"	// ADD-BY-LEETEN 03/28/2012

extern CUDPPHandle cudpp;
extern bool bIsUsingCpu;
extern bool bIsPrintingTiming;

size_t uNrOfPoints1;
size_t uNrOfPoints2;

float* pfMinDistFromPoint1ToPoints2_device;
float *pfCompactedMinDistFromPoint1ToPoints2_device;
size_t *puNonUsed_device;
unsigned int *puSegBegin_device;
float *pfDist1To2_device;
float *pfDist1To2_host;
float4 *pf4Points1_device;
float4 *pf4Points2_device;
float* pfDists_device;

static bool bIsFreeSet;

void
_GPUHausFree()
{
	FREE_MEMORY(pf4Points1_device);
	FREE_MEMORY(pf4Points2_device);
	FREE_MEMORY(pfDists_device);
	FREE_MEMORY(pfDist1To2_device);
	FREE_MEMORY_ON_HOST(pfDist1To2_host);
	FREE_MEMORY(puSegBegin_device);
	FREE_MEMORY(pfMinDistFromPoint1ToPoints2_device);
	FREE_MEMORY(pfCompactedMinDistFromPoint1ToPoints2_device);
	FREE_MEMORY(puNonUsed_device);
	ASSERT_OR_LOG(
		!cReduceMinDistFromPoint1ToPoints2.hPlan ||
		CUDPP_SUCCESS == cudppDestroyPlan(cReduceMinDistFromPoint1ToPoints2.hPlan), "");
	ASSERT_OR_LOG(
		!cFindMinDistFromPoint1ToPoints2.hPlan ||
		CUDPP_SUCCESS == cudppDestroyPlan(cFindMinDistFromPoint1ToPoints2.hPlan), "");
	ASSERT_OR_LOG(
		!cCompactMinDistFromPoint1ToPoints2.hPlan || 
		CUDPP_SUCCESS == cudppDestroyPlan(cCompactMinDistFromPoint1ToPoints2.hPlan), "");
}

void
_GPUHausSetLengths
(
	size_t uNrOfPoints1,
	size_t uNrOfPoints2
)
{
	::uNrOfPoints1 = uNrOfPoints1;
	::uNrOfPoints2 = uNrOfPoints2;
	if( !bIsFreeSet )
	{
		atexit(_GPUHausFree);
		bIsFreeSet  = true;
	}

	_GPUHausFree();

	// pass 2: compute the min per segment
	// create the segments on both host and device sides
	#if	IS_MEMCPY_SEG
	TBuffer<unsigned int> puSegBegin;
	puSegBegin.alloc(uNrOfPoints1 * uNrOfPoints2);
	for(size_t p1 = 0; p1 < uNrOfPoints1; p1++)
		puSegBegin[p1 * uNrOfPoints2] = 1;

	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&puSegBegin_device,
			puSegBegin.USize() * sizeof(puSegBegin_device[0]) ) );
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMemcpy(
			&puSegBegin_device[0],
			&puSegBegin[0],
			puSegBegin.USize() * sizeof(puSegBegin_device[0]),
			cudaMemcpyHostToDevice) );
	#else	// #if	IS_MEMCPY_SEG
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&puSegBegin_device,
			uNrOfPoints1 * uNrOfPoints2 * sizeof(puSegBegin_device[0]) ) );

	CUDA_SAFE_CALL_NO_SYNC(
		cudaMemset(
			puSegBegin_device,
			0,
			uNrOfPoints1 * uNrOfPoints2 * sizeof(puSegBegin_device[0]) ) );
	dim3 v3Blk = dim3(BLOCK_DIM_X * BLOCK_DIM_Y);
	size_t uNrOfNeededThreads = uNrOfPoints1 * uNrOfPoints1;
	size_t uNrOfBlocks = (unsigned int)ceilf((float)uNrOfNeededThreads / (float)v3Blk.x);
	dim3 v3Grid = dim3(
		min(uNrOfBlocks, (size_t)GRID_DIM_X),
		(unsigned int)ceil((double)uNrOfBlocks / (double)GRID_DIM_X)
		);
	_SetupSegments_kernel<<<v3Grid, v3Blk, 0>>>
	(
		uNrOfPoints1,
		uNrOfPoints2,
		puSegBegin_device
	);
	CUT_CHECK_ERROR("_SetupSegments_kernel() failed");
	#endif	// #if	IS_MEMCPY_SEG

	// create the cudpp plans
	cFindMinDistFromPoint1ToPoints2.cConfig.op =		CUDPP_MIN;
	cFindMinDistFromPoint1ToPoints2.cConfig.algorithm =	CUDPP_SEGMENTED_SCAN;
	cFindMinDistFromPoint1ToPoints2.cConfig.datatype =	CUDPP_FLOAT;
	cFindMinDistFromPoint1ToPoints2.cConfig.options =	CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
	ASSERT_OR_LOG(CUDPP_SUCCESS == 
		cudppPlan(
			cudpp,
			&cFindMinDistFromPoint1ToPoints2.hPlan, 
			cFindMinDistFromPoint1ToPoints2.cConfig, 
			uNrOfPoints1 * uNrOfPoints2, 
			1, 
			0), 
		"");

	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pfMinDistFromPoint1ToPoints2_device,
			uNrOfPoints1 * uNrOfPoints2 * sizeof(pfMinDistFromPoint1ToPoints2_device[0]) ) );

	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pfCompactedMinDistFromPoint1ToPoints2_device,
			uNrOfPoints1 * sizeof(pfCompactedMinDistFromPoint1ToPoints2_device[0]) ) );

	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&puNonUsed_device,
			uNrOfPoints1 * sizeof(puNonUsed_device[0]) ) );

	cCompactMinDistFromPoint1ToPoints2.cConfig.algorithm =	CUDPP_COMPACT;
	cCompactMinDistFromPoint1ToPoints2.cConfig.datatype =	CUDPP_FLOAT;
	cCompactMinDistFromPoint1ToPoints2.cConfig.options =	CUDPP_OPTION_FORWARD;
	ASSERT_OR_LOG(CUDPP_SUCCESS == 
		cudppPlan(
			cudpp,
			&cCompactMinDistFromPoint1ToPoints2.hPlan, 
			cCompactMinDistFromPoint1ToPoints2.cConfig, 
			uNrOfPoints1 * uNrOfPoints2, 1, 0), 
		"");

	cReduceMinDistFromPoint1ToPoints2.cConfig.op =		CUDPP_MAX;
	cReduceMinDistFromPoint1ToPoints2.cConfig.algorithm =	CUDPP_REDUCE;
	cReduceMinDistFromPoint1ToPoints2.cConfig.datatype =	CUDPP_FLOAT;
	cReduceMinDistFromPoint1ToPoints2.cConfig.options =	CUDPP_OPTION_FORWARD;
	ASSERT_OR_LOG(CUDPP_SUCCESS == 
		cudppPlan(
			cudpp,
			&cReduceMinDistFromPoint1ToPoints2.hPlan, 
			cReduceMinDistFromPoint1ToPoints2.cConfig, 
			uNrOfPoints1, 1, 0), 
		"");

	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pfDist1To2_device,
			sizeof(pfDist1To2_device[0]) ) );
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMallocHost(
			&pfDist1To2_host,
			sizeof(pfDist1To2_host[0]) ) );

	// allocate a 1D linear buffer for uNrOfPoints1 * uNrOfPoints2
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pfDists_device,
			uNrOfPoints1 * uNrOfPoints2 * sizeof(pfDists_device[0]) ) );
}

void
_GPUHausSetPoints
(
	int iIndex,
	size_t uNrOfPoints,
	float4 pf4Points[]
)
{
	ASSERT_OR_LOG(iIndex < 2 && 
		((0 == iIndex && ::uNrOfPoints1 == uNrOfPoints) ||
		 (1 == iIndex && ::uNrOfPoints2 == uNrOfPoints) ), "");

	switch(iIndex)
	{
	case 0:
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMalloc(
				&pf4Points1_device,
				uNrOfPoints * sizeof(pf4Points1_device[0]) ) );
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				&pf4Points1_device[0],
				&pf4Points[0],
				uNrOfPoints * sizeof(pf4Points1_device[0]),
				cudaMemcpyHostToDevice) );
		SETUP_TEXTURE(
			t1Df4Points1, 
			cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp, 
			cudaFilterModePoint, false);
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaBindTexture(
				NULL, 
				t1Df4Points1, 
				&pf4Points1_device[0],
				t1Df4Points1.channelDesc,
				uNrOfPoints1 * sizeof(pf4Points1_device[0])));
		break;
	case 1:
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMalloc(
				&pf4Points2_device,
				uNrOfPoints * sizeof(pf4Points2_device[0]) ) );
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				&pf4Points2_device[0],
				&pf4Points[0],
				uNrOfPoints * sizeof(pf4Points2_device[0]),
				cudaMemcpyHostToDevice) );
		SETUP_TEXTURE(
			t1Df4Points2, 
			cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp, 
			cudaFilterModePoint, false);
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaBindTexture(
				NULL, 
				t1Df4Points2, 
				&pf4Points2_device[0],
				t1Df4Points2.channelDesc,
				uNrOfPoints2 * sizeof(pf4Points2_device[0])));
		break;
	}
}

void
_GPUHausCompByCpu
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	float* pfDist1To2
)
{
	ASSERT_OR_LOG(::uNrOfPoints1 == uNrOfPoints1 && ::uNrOfPoints2 == uNrOfPoints2, "");
LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);
LIBCLOCK_BEGIN(bIsPrintingTiming);
	
	float fDist1To2 = -(float)HUGE_VAL;
	for(size_t p1 = 0; p1 < uNrOfPoints1; p1++)
	{
		float fMinDist = (float)HUGE_VAL;
		for(size_t p2 = 0; p2 < uNrOfPoints2; p2++)
		{
			float fDx = pf4Points2[p2].x - pf4Points1[p1].x;
			float fDy = pf4Points2[p2].y - pf4Points1[p1].y;
			float fDz = pf4Points2[p2].z - pf4Points1[p1].z;
			float fDist = fDx * fDx + fDy * fDy + fDz * fDz;
			fMinDist = min(fMinDist, fDist);
		}
		fDist1To2 = max(fDist1To2, fMinDist);
	}
	*pfDist1To2 = sqrtf(fDist1To2);
LIBCLOCK_END(bIsPrintingTiming);
LIBCLOCK_PRINT(bIsPrintingTiming);
}


void
_GPUHausComp
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	float* pfDist1To2
)
{
	ASSERT_OR_LOG(::uNrOfPoints1 == uNrOfPoints1 && ::uNrOfPoints2 == uNrOfPoints2, "");
	if( bIsUsingCpu )
	{
		_GPUHausCompByCpu
		(
			uNrOfPoints1,
			pf4Points1,

			uNrOfPoints2,
			pf4Points2,

			pfDist1To2
		);

		return;
	}

LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);

LIBCLOCK_BEGIN(bIsPrintingTiming);
	// pass 1: compute pair-wise distance
	dim3 v3Blk = dim3(BLOCK_DIM_X * BLOCK_DIM_Y);
	size_t uNrOfNeededThreads = uNrOfPoints1 * uNrOfPoints2;
	size_t uNrOfBlocks = (unsigned int)ceilf((float)uNrOfNeededThreads / (float)v3Blk.x);
	dim3 v3Grid = dim3(
		min(uNrOfBlocks, (size_t)GRID_DIM_X),
		(unsigned int)ceil((double)uNrOfBlocks / (double)GRID_DIM_X)
		);
	_CompPairDist_kernel<<<v3Grid, v3Blk, 0>>>
	(
		uNrOfPoints1,
		uNrOfPoints2,
		&pfDists_device[0]
	);	
	CUT_CHECK_ERROR("_CompPairDist_kernel() failed");
LIBCLOCK_END(bIsPrintingTiming);

	#if		IS_COMPACT_AND_REDUCE_ON_CPU	
LIBCLOCK_BEGIN(bIsPrintingTiming);
	TBuffer<float> pfDists;
	pfDists.alloc(uNrOfPoints1 * uNrOfPoints2);

	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMemcpy(
			&pfDists[0],
			&pfDists_device[0],
			pfDists.USize() * sizeof(pfDists[0]),
			cudaMemcpyDeviceToHost) );

	float fDist1To2 = -(float)HUGE_VAL;
	for(size_t p = 0,	p1 = 0; p1 < uNrOfPoints1; p1++)
	{
		float fMinDist = (float)HUGE_VAL;
		for(size_t	p2 = 0; p2 < uNrOfPoints2; p2++, p++)
			fMinDist = min(fMinDist, pfDists[p]);
		fDist1To2 = max(fDist1To2, fMinDist);
	}
LIBCLOCK_END(bIsPrintingTiming);
	#else	// #if	IS_COMPACT_AND_REDUCE_ON_CPU	
	// pass 2: compute the min per segment
LIBCLOCK_BEGIN(bIsPrintingTiming);
	ASSERT_OR_LOG(CUDPP_SUCCESS == 
		cudppSegmentedScan(
			cFindMinDistFromPoint1ToPoints2.hPlan,
			pfMinDistFromPoint1ToPoints2_device,
			pfDists_device,
			puSegBegin_device,	
			uNrOfPoints1 * uNrOfPoints2), 
		"");
	CUT_CHECK_ERROR("cudppSegmentedScan() failed");
LIBCLOCK_END(bIsPrintingTiming);

	// pass 3: compact the min
LIBCLOCK_BEGIN(bIsPrintingTiming);
	ASSERT_OR_LOG(CUDPP_SUCCESS == 
		cudppCompact(
			cCompactMinDistFromPoint1ToPoints2.hPlan,
			pfCompactedMinDistFromPoint1ToPoints2_device,
			puNonUsed_device,
			pfMinDistFromPoint1ToPoints2_device,
			puSegBegin_device,	
			uNrOfPoints1 * uNrOfPoints2),
		"");
	CUT_CHECK_ERROR("cudppCompact() failed");
LIBCLOCK_END(bIsPrintingTiming);

	float fDist1To2;
	#if		IS_REDUCE_ON_CPU
LIBCLOCK_BEGIN(bIsPrintingTiming);
	TBuffer<float> pfMinDistFromPoint1ToPoints2;
	pfMinDistFromPoint1ToPoints2.alloc(uNrOfPoints1);
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMemcpy(
			&pfMinDistFromPoint1ToPoints2[0],
			&pfCompactedMinDistFromPoint1ToPoints2_device[0],
			uNrOfPoints1 * sizeof(pfMinDistFromPoint1ToPoints2[0]),
			cudaMemcpyDeviceToHost) );
	fDist1To2 = -(float)HUGE_VAL;
	for(size_t p1 = 0; p1 < pfMinDistFromPoint1ToPoints2.USize(); p1++)
		fDist1To2 = max(fDist1To2, pfMinDistFromPoint1ToPoints2[p1]);
LIBCLOCK_END(bIsPrintingTiming);
	#else	// #if	IS_REDUCE_ON_CPU
LIBCLOCK_BEGIN(bIsPrintingTiming);
	// pass 4: reduce the min distance with max. operator to get the Hausdorff distance
	ASSERT_OR_LOG(CUDPP_SUCCESS == 
		cudppReduce(
			cReduceMinDistFromPoint1ToPoints2.hPlan,
			pfDist1To2_device,
			pfCompactedMinDistFromPoint1ToPoints2_device,
			uNrOfPoints1),
		"");
	CUT_CHECK_ERROR("cudppReduce() failed");
LIBCLOCK_END(bIsPrintingTiming);

LIBCLOCK_BEGIN(bIsPrintingTiming);
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMemcpy(
			&pfDist1To2_host[0],
			&pfDist1To2_device[0],
			sizeof(pfDist1To2_host[0]),
			cudaMemcpyDeviceToHost) );
	fDist1To2 = pfDist1To2_host[0];
LIBCLOCK_END(bIsPrintingTiming);
	#endif	// #if	IS_REDUCE_ON_CPU
	#endif	// #if	IS_COMPACT_AND_REDUCE_ON_CPU	
	*pfDist1To2 = sqrtf(fDist1To2);
LIBCLOCK_PRINT(bIsPrintingTiming);
}

/*

$Log$

*/