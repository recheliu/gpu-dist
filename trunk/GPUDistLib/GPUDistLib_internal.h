#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <assert.h>	// ADD-BY-LEETEN 03/28/2012

#if	defined(WITH_CUDPP)
	#include "cudpp.h"
#endif

#ifdef	 WIN32
	#undef	 WIN32
	#include "liblog.h"
	#define  WIN32
#else
	#include "liblog.h"
#endif
#include "cuda_macro.h"	
// DEL-BY-LEETEN 04/09/2012:	#include "cuda_macro3d.h"	
#include "libclock.h"
#include "libbuf.h"	

#include "GPUDistLib.h"	

#define BLOCK_DIM_X	16
#define BLOCK_DIM_Y	8
#define GRID_DIM_X	128
#define GRID_DIM_Y	128
#define BATCH_SIZE	(BLOCK_DIM_X * BLOCK_DIM_Y * GRID_DIM_X * GRID_DIM_Y)

// ADD-BY-LEETEN 04/07/2012-BEGIN
//! Decide whether the squared root is computed
#define IS_SQRT		0
// ADD-BY-LEETEN 04/07/2012-END

// ADD-BY-LEETEN 04/07/2012-BEGIN
//! Decide whether multiple points are computed in a single kernel call
#define IS_COMP_MULTI_POINTS	1
#if	IS_COMP_MULTI_POINTS
	#define MAX_NR_OF_COMP_POINTS	128
#endif	// #if	IS_COMP_MULTI_POINTS
// ADD-BY-LEETEN 04/07/2012-END

#if	defined(WITH_CUDPP)
struct CConfigPlan
{
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
};

extern CUDPPHandle cudpp;
#endif	// #if	defined(WITH_CUDPP)

extern bool bIsUsingCpu;
extern bool bIsPrintingTiming;

/*

$Log$

*/
