#include "GPUDistLib_internal.h"

#define WARP_SIZE				256		// 128
#define NR_OF_POINTS1_PER_BATCH			WARP_SIZE
#define POINT2_TEX_WIDTH			4096

#define POINT_AS_CONSTANT		1
#define POINT_AS_LINEAR_BUFFER		2
#define POINT_AS_ARRAY			3
#define POINT1_AS			POINT_AS_ARRAY
#define POINT2_AS			POINT_AS_ARRAY

#include "kernel_SetupSegBegin.h"	
#include "kernel_CompPairDistInBatch.h"	

static size_t uNrOfBatchesForPoints2;
static size_t uNrOfPoints2PerBatch;
static size_t uNrOfPoints2LastBatch;
static size_t uNrOfPoints1PerBatch;
static size_t uNrOfPoints1LastBatch;
static size_t uNrOfBatchesForPoints1;
static size_t uNrOfPointsPerBatch;
static size_t uNrOFPoints1;
static size_t uNrOFPoints2;
static size_t uPoint2TexWidth;
static size_t uPoint2TexHeight;

// cudpp plans
static CConfigPlan
		cFindMinDistFromPoint1ToPoints2, 
		cCompactMinDistFromPoint1ToPoints2;

// linear buffers on the device side
static float *pfDistsInBatch_device;
static unsigned int *puSegBeginPerBatch_device;
static unsigned int *puSegBeginLastBatch_device;
#if	POINT1_AS == POINT_AS_LINEAR_BUFFER
	static float4 *pf4Points1InBatch_device;
#elif	POINT1_AS == POINT_AS_ARRAY
	static cudaArray* pf4Points1InBatch_array;
#endif

#if	POINT2_AS == POINT_AS_LINEAR_BUFFER
	static float4 *pf4Points2InBatch_device;
#elif	POINT2_AS == POINT_AS_ARRAY
	static cudaArray* pf4Points2InBatch_array;
#endif

static 	float* pfMinDistFromPoint1ToPoints2_device;
static 	float *ppfCompactedMinDistFromPoint1ToPoints2_device[2];
static 	size_t *puNonUsed_device;

void
_cuDistFree
(
 )
{
	FREE_MEMORY(pfDistsInBatch_device);
	FREE_MEMORY(puSegBeginPerBatch_device);
	FREE_MEMORY(puSegBeginLastBatch_device);
	#if	POINT1_AS	==	POINT_AS_LINEAR_BUFFER
		FREE_MEMORY(pf4Points1InBatch_device);
	#elif	POINT1_AS	==	POINT_AS_ARRAY
		FREE_ARRAY(pf4Points1InBatch_array);
	#endif
	#if	POINT2_AS	==	POINT_AS_LINEAR_BUFFER
		FREE_MEMORY(pf4Points2InBatch_device);
	#elif	POINT2_AS	==	POINT_AS_ARRAY
		FREE_ARRAY(pf4Points2InBatch_array);
	#endif
	FREE_MEMORY(pfMinDistFromPoint1ToPoints2_device);
	for(int i = 0; i < 2; i++)
		FREE_MEMORY(ppfCompactedMinDistFromPoint1ToPoints2_device[i]);
	FREE_MEMORY(puNonUsed_device);
	ASSERT_OR_LOG(
		!cFindMinDistFromPoint1ToPoints2.hPlan ||
		CUDPP_SUCCESS == cudppDestroyPlan(cFindMinDistFromPoint1ToPoints2.hPlan), "");
	ASSERT_OR_LOG(
		!cCompactMinDistFromPoint1ToPoints2.hPlan || 
		CUDPP_SUCCESS == cudppDestroyPlan(cCompactMinDistFromPoint1ToPoints2.hPlan), "");
}

void
_cuDistInit
(
	size_t uBatchSize
)
{
	atexit(_cuDistFree);

	// allocate a space for one batch
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pfDistsInBatch_device,
			uBatchSize * sizeof(pfDistsInBatch_device[0]) ) );

	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&puSegBeginPerBatch_device,
			uBatchSize * sizeof(puSegBeginPerBatch_device[0]) ) );

	#if	POINT1_AS	==	POINT_AS_LINEAR_BUFFER
	////////////////// setup point1/batch as a 1D texture /////////////////////////////////////
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pf4Points1InBatch_device,
			NR_OF_POINTS1_PER_BATCH * sizeof(pf4Points1InBatch_device[0]) ) );
	SETUP_TEXTURE(
		t1Df4Points1InBatch, 
		cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp, 
		cudaFilterModePoint, false);
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaBindTexture(
			NULL, 
			t1Df4Points1InBatch, 
			&pf4Points1InBatch_device[0],
			t1Df4Points1InBatch.channelDesc,
			NR_OF_POINTS1_PER_BATCH * sizeof(pf4Points1InBatch_device[0])));

	#elif	POINT1_AS	==	POINT_AS_ARRAY
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMallocArray(
			&pf4Points1InBatch_array, 
			&t1Df4Points1InBatch.channelDesc, 
			NR_OF_POINTS1_PER_BATCH, 
			1) );
	SETUP_TEXTURE(
		t1Df4Points1InBatch, 
		cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp, 
		cudaFilterModePoint, false);
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaBindTextureToArray(
			t1Df4Points1InBatch, 
			pf4Points1InBatch_array, 
			t1Df4Points1InBatch.channelDesc) );
	#endif

	////////////////////////// create cudpp plans ///////////////////////////
	// create a plan for segmented scanning
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pfMinDistFromPoint1ToPoints2_device,
			uBatchSize * sizeof(pfMinDistFromPoint1ToPoints2_device[0]) ) );
	cFindMinDistFromPoint1ToPoints2.cConfig.op =		CUDPP_MIN;
	cFindMinDistFromPoint1ToPoints2.cConfig.algorithm =	CUDPP_SEGMENTED_SCAN;
	cFindMinDistFromPoint1ToPoints2.cConfig.datatype =	CUDPP_FLOAT;
	cFindMinDistFromPoint1ToPoints2.cConfig.options =	CUDPP_OPTION_BACKWARD | CUDPP_OPTION_INCLUSIVE;
	ASSERT_OR_LOG(CUDPP_SUCCESS == 
		cudppPlan(
			cudpp,
			&cFindMinDistFromPoint1ToPoints2.hPlan, 
			cFindMinDistFromPoint1ToPoints2.cConfig, 
			uBatchSize, 
			1, 
			0), 
		"");

	// create a plan for compacting
	for(int i = 0; i < 2; i++)
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMalloc(
				&ppfCompactedMinDistFromPoint1ToPoints2_device[i],
				NR_OF_POINTS1_PER_BATCH * sizeof(ppfCompactedMinDistFromPoint1ToPoints2_device[i][0]) ) );
	SETUP_TEXTURE(
		t1DfDistsInBatch, 
		cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp, 
		cudaFilterModePoint, false);

	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&puNonUsed_device,
			NR_OF_POINTS1_PER_BATCH * sizeof(puNonUsed_device[0]) ) );
	cCompactMinDistFromPoint1ToPoints2.cConfig.algorithm =	CUDPP_COMPACT;
	cCompactMinDistFromPoint1ToPoints2.cConfig.datatype =	CUDPP_FLOAT;
	cCompactMinDistFromPoint1ToPoints2.cConfig.options =	CUDPP_OPTION_FORWARD;
	ASSERT_OR_LOG(CUDPP_SUCCESS == 
		cudppPlan(
			cudpp,
			&cCompactMinDistFromPoint1ToPoints2.hPlan, 
			cCompactMinDistFromPoint1ToPoints2.cConfig, 
			uBatchSize, 
			1, 
			0), 
		"");
}

void
_cuDistSetLengths
(
	size_t uNrOfPoints1,
	size_t uNrOfPoints2
)
{
	/////////////////////////////////////////////////////
	// decide #batches and number of points in the batch
	uNrOfPoints1PerBatch = min((size_t)NR_OF_POINTS1_PER_BATCH, uNrOfPoints1);
	uNrOfBatchesForPoints1 = (size_t)ceil((double)uNrOfPoints1 / (double)uNrOfPoints1PerBatch);
	uNrOfPoints2PerBatch = (size_t)floor((double)BATCH_SIZE / (double)uNrOfPoints1PerBatch);
	uNrOfPoints2PerBatch = WARP_SIZE * (size_t)floor((double)uNrOfPoints2PerBatch / (double)WARP_SIZE);
	uNrOfPoints2PerBatch = min(uNrOfPoints2PerBatch, uNrOfPoints2);
	uNrOfBatchesForPoints2 = (size_t)ceil((double)uNrOfPoints2 / (double)uNrOfPoints2PerBatch);
	uNrOfPointsPerBatch = uNrOfBatchesForPoints1 * uNrOfBatchesForPoints2;
	uNrOfPoints1LastBatch = uNrOfPoints1 % uNrOfPoints1PerBatch;
	uNrOfPoints2LastBatch = uNrOfPoints2 % uNrOfPoints2PerBatch;

	LOG_VAR(uNrOfPoints1);
	LOG_VAR(uNrOfPoints1PerBatch);
	LOG_VAR(uNrOfPoints1LastBatch);
	LOG_VAR(uNrOfBatchesForPoints1);
	LOG_VAR(uNrOfPoints2);
	LOG_VAR(uNrOfPoints2PerBatch);
	LOG_VAR(uNrOfPoints2LastBatch);
	LOG_VAR(uNrOfBatchesForPoints2);

	// init. the batch mode
	_cuDistInit(uNrOfPoints1PerBatch * uNrOfPoints2PerBatch);

	////////////////// setup point2/batch as a 2D texture /////////////////////////////////////
	// setup the texture
	uPoint2TexWidth = POINT2_TEX_WIDTH;
	uPoint2TexHeight = (size_t)ceil( (double)uNrOfPoints2PerBatch / (double)uPoint2TexWidth );

	#if	POINT2_AS	== POINT_AS_LINEAR_BUFFER
	// allocate a linear buffers to hold the 
	FREE_MEMORY(pf4Points2InBatch_device);
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMalloc(
			&pf4Points2InBatch_device,
			uNrOfPoints2PerBatch * sizeof(pf4Points2InBatch_device[0]) ) );

	SETUP_TEXTURE(
		t2Df4Points2InBatch, 
		cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp, 
		cudaFilterModePoint, false);
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaBindTexture2D(
			NULL, 
			t2Df4Points2InBatch, 
			&pf4Points2InBatch_device[0],
			t2Df4Points2InBatch.channelDesc,
			uPoint2TexWidth,
			uPoint2TexHeight,
			uPoint2TexWidth * sizeof(pf4Points2InBatch_device[0])));
	#elif	POINT2_AS	== POINT_AS_ARRAY
	FREE_ARRAY(pf4Points2InBatch_array);
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaMallocArray(
			&pf4Points2InBatch_array, 
			&t2Df4Points2InBatch.channelDesc, 
			uPoint2TexWidth, 
			uPoint2TexHeight) );
	SETUP_TEXTURE(
		t2Df4Points2InBatch, 
		cudaAddressModeClamp, cudaAddressModeClamp, cudaAddressModeClamp, 
		cudaFilterModePoint, false);
	CUDA_SAFE_CALL_NO_SYNC( 
		cudaBindTextureToArray(
			t2Df4Points2InBatch, 
			pf4Points2InBatch_array, 
			t2Df4Points2InBatch.channelDesc) );
	#endif

		
	///////////////////////// create segments /////////////////////
	// setup the regular batch
	{
		CUDA_SAFE_CALL_NO_SYNC(
			cudaMemset(
				puSegBeginPerBatch_device,
				0,
				uNrOfPoints1PerBatch * uNrOfPoints2PerBatch * sizeof(puSegBeginPerBatch_device[0]) ) );

		dim3 v3Blk = dim3(BLOCK_DIM_X * BLOCK_DIM_Y);
		size_t uNrOfNeededThreads = uNrOfPoints1PerBatch * uNrOfPoints2PerBatch;
		size_t uNrOfBlocks = (unsigned int)ceilf((float)uNrOfNeededThreads / (float)v3Blk.x);
		dim3 v3Grid = dim3(
			min(uNrOfBlocks, (size_t)GRID_DIM_X),
			(unsigned int)ceil((double)uNrOfBlocks / (double)GRID_DIM_X)
			);
		_SetupSegments_kernel<<<v3Grid, v3Blk, 0>>>
		(
			uNrOfPoints1PerBatch,
			uNrOfPoints2PerBatch,
			puSegBeginPerBatch_device
		);
		CUT_CHECK_ERROR("_SetupSegments_kernel() failed");
	}

	// setup the last batch
	FREE_MEMORY(puSegBeginLastBatch_device);
	if( uNrOfPoints2LastBatch )
	{
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMalloc(
				&puSegBeginLastBatch_device,
				uNrOfPoints2LastBatch * uNrOfPoints1PerBatch * sizeof(puSegBeginLastBatch_device[0]) ) );

		CUDA_SAFE_CALL_NO_SYNC(
			cudaMemset(
				puSegBeginLastBatch_device,
				0,
				uNrOfPoints1PerBatch * uNrOfPoints2LastBatch * sizeof(puSegBeginLastBatch_device[0]) ) );
		dim3 v3Blk = dim3(BLOCK_DIM_X * BLOCK_DIM_Y);
		size_t uNrOfNeededThreads = uNrOfPoints1PerBatch * uNrOfPoints2LastBatch ;
		size_t uNrOfBlocks = (unsigned int)ceilf((float)uNrOfNeededThreads / (float)v3Blk.x);
		dim3 v3Grid = dim3(
			min(uNrOfBlocks, (size_t)GRID_DIM_X),
			(unsigned int)ceil((double)uNrOfBlocks / (double)GRID_DIM_X)
			);
		_SetupSegments_kernel<<<v3Grid, v3Blk, 0>>>
		(
			uNrOfPoints1PerBatch,
			uNrOfPoints2LastBatch,
			puSegBeginLastBatch_device
		);
		CUT_CHECK_ERROR("_SetupSegments_kernel() failed");
	}
}

void
_cuDistComp
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	float pfDists[]
)
{
	if( 1 == uNrOfBatchesForPoints2 )
	{
		// upload all points
#if	POINT2_AS	== POINT_AS_LINEAR_BUFFER
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				&pf4Points2InBatch_device[0],
				&pf4Points2[0],
				uNrOfPoints2 * sizeof(pf4Points2InBatch_device[0]),
				cudaMemcpyHostToDevice) );
#elif	POINT2_AS	== POINT_AS_ARRAY
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpyToArray(
				pf4Points2InBatch_array, 
				0, 
				0, 
				&pf4Points2[0], 
				uNrOfPoints2 * sizeof(pf4Points2[0]), 
				cudaMemcpyHostToDevice) );
#endif
	}

	for(size_t		b1 = 0; b1 < uNrOfBatchesForPoints1; b1++)
	{
		size_t uNrOfPoints1CurrentBatch = 
			(uNrOfPoints1LastBatch > 0 && b1 == uNrOfBatchesForPoints1-1)?
				uNrOfPoints1LastBatch:
				uNrOfPoints1PerBatch;

		// upload the points1 in batch b1
		#if	POINT1_AS	==	POINT_AS_LINEAR_BUFFER
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				&pf4Points1InBatch_device[0],
				&pf4Points1[b1 * uNrOfPoints1PerBatch],
				uNrOfPoints1CurrentBatch * sizeof(pf4Points1InBatch_device[0]),
				cudaMemcpyHostToDevice) );
		#elif	POINT1_AS	==	POINT_AS_ARRAY
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpyToArray(
				pf4Points1InBatch_array, 
				0, 
				0, 
				&pf4Points1[b1 * uNrOfPoints1PerBatch], 
				uNrOfPoints1CurrentBatch * sizeof(pf4Points1[0]), 
				cudaMemcpyHostToDevice) );
		#elif	POINT1_AS	==	POINT_AS_CONSTANT
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpyToSymbol(
				"pf4Points1InBatch_const",	
				&pf4Points1[b1 * uNrOfPoints1PerBatch],	
				sizeof(pf4Points1InBatch_const[0]) * uNrOfPoints1CurrentBatch,	
				0, 
				cudaMemcpyHostToDevice) );
		#endif

		for(size_t	b2 = 0; b2 < uNrOfBatchesForPoints2; b2++)
		{
LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);
LIBCLOCK_BEGIN(bIsPrintingTiming);
			size_t uNrOfPoints2CurrentBatch;
			unsigned int *puSegBegin_device;
			if( uNrOfPoints2LastBatch > 0 && b2 == uNrOfBatchesForPoints2-1 ) 
			{
				uNrOfPoints2CurrentBatch = uNrOfPoints2LastBatch;
				puSegBegin_device = puSegBeginLastBatch_device;
			}
			else
			{
				uNrOfPoints2CurrentBatch = uNrOfPoints2PerBatch;
				puSegBegin_device = puSegBeginPerBatch_device;
			}

			if( 1 < uNrOfBatchesForPoints2 )
			{
#if	POINT2_AS	== POINT_AS_LINEAR_BUFFER
				// upload the points2 in batch b2
				CUDA_SAFE_CALL_NO_SYNC( 
					cudaMemcpy(
						&pf4Points2InBatch_device[0],
						&pf4Points2[b2 * uNrOfPoints2PerBatch],
						uNrOfPoints2CurrentBatch * sizeof(pf4Points2InBatch_device[0]),
						cudaMemcpyHostToDevice) );
#elif	POINT2_AS	== POINT_AS_ARRAY
			CUDA_SAFE_CALL_NO_SYNC( 
				cudaMemcpyToArray(
					pf4Points2InBatch_array, 
					0, 
					0, 
					&pf4Points2[b2 * uNrOfPoints2PerBatch], 
					uNrOfPoints2CurrentBatch * sizeof(pf4Points2[0]), 
					cudaMemcpyHostToDevice) );
#endif

				CUDA_SAFE_CALL_NO_SYNC( 
					cudaBindTexture(
						NULL, 
						t1DfDistsInBatch, 
						&ppfCompactedMinDistFromPoint1ToPoints2_device[1 - b2 % 2][0],
						t1DfDistsInBatch.channelDesc,
						uNrOfPoints1CurrentBatch * sizeof(ppfCompactedMinDistFromPoint1ToPoints2_device[0][0])));
			}
LIBCLOCK_END(bIsPrintingTiming);

			//////////// pass 1: compute pair-wise distance /////////////////
			// invoke the kernel to compute pair wise distance
			// #points1/batch, #points/batch
			// pass 1: compute pair-wise distance
LIBCLOCK_BEGIN(bIsPrintingTiming);
			dim3 v3Blk = dim3(BLOCK_DIM_X * BLOCK_DIM_Y);
			size_t uNrOfNeededThreads = uNrOfPoints1CurrentBatch * uNrOfPoints2CurrentBatch;
			size_t uNrOfBlocks = (unsigned int)ceilf((float)uNrOfNeededThreads / (float)v3Blk.x);
			dim3 v3Grid = dim3(
				min(uNrOfBlocks, (size_t)GRID_DIM_X),
				(unsigned int)ceil((double)uNrOfBlocks / (double)GRID_DIM_X)
				);
			_CompPairDistInBatch_kernel<<<v3Grid, v3Blk, 0>>>
			(
				b2,
				uNrOfPoints1CurrentBatch,
				uNrOfPoints2CurrentBatch,
				uPoint2TexWidth,
				&pfDistsInBatch_device[0]
			);	
			CUT_CHECK_ERROR("_CompPairDist_kernel() failed");
LIBCLOCK_END(bIsPrintingTiming);

			//////////// pass 2: do segmented scan /////////////////
LIBCLOCK_BEGIN(bIsPrintingTiming);
			ASSERT_OR_LOG(CUDPP_SUCCESS == 
				cudppSegmentedScan(
					cFindMinDistFromPoint1ToPoints2.hPlan,
					pfMinDistFromPoint1ToPoints2_device,
					pfDistsInBatch_device,
					puSegBegin_device,	
					uNrOfPoints1CurrentBatch * uNrOfPoints2CurrentBatch), 
				"");
			CUT_CHECK_ERROR("cudppSegmentedScan() failed");
LIBCLOCK_END(bIsPrintingTiming);

			//////////// pass 3: compact the scanned result /////////////////
LIBCLOCK_BEGIN(bIsPrintingTiming);
			ASSERT_OR_LOG(CUDPP_SUCCESS == 
				cudppCompact(
					cCompactMinDistFromPoint1ToPoints2.hPlan,
					ppfCompactedMinDistFromPoint1ToPoints2_device[b2 % 2],
					puNonUsed_device,
					pfMinDistFromPoint1ToPoints2_device,
					puSegBegin_device,	
					uNrOfPoints1CurrentBatch * uNrOfPoints2CurrentBatch),
				"");
			CUT_CHECK_ERROR("cudppCompact() failed");
LIBCLOCK_END(bIsPrintingTiming);
LIBCLOCK_PRINT(bIsPrintingTiming);
		} // b2

		// download the result back
		CUDA_SAFE_CALL_NO_SYNC( 
			cudaMemcpy(
				&pfDists[b1 * uNrOfPoints1PerBatch],
				&ppfCompactedMinDistFromPoint1ToPoints2_device[1 - uNrOfBatchesForPoints2 % 2][0],
				uNrOfPoints1CurrentBatch * sizeof(ppfCompactedMinDistFromPoint1ToPoints2_device[0][0]),
				cudaMemcpyDeviceToHost) );
	}
}

/*

$Log$

*/