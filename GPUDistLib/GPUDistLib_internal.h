#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#include <assert.h>	

#if	defined(WITH_CUDPP)
	#include "cudpp.h"
#endif

#include <lib3ds/vector.h>	

#ifdef	 WIN32
	#undef	 WIN32
	#include "liblog.h"
	#define  WIN32
#else
	#include "liblog.h"
#endif
#include "cuda_macro.h"	
#include "libclock.h"
#include "libbuf.h"	

#include "GPUDistLib.h"	

#define BLOCK_DIM_X	16
#define BLOCK_DIM_Y	8
#define GRID_DIM_X	128
#define GRID_DIM_Y	128
#define BATCH_SIZE	(BLOCK_DIM_X * BLOCK_DIM_Y * GRID_DIM_X * GRID_DIM_Y)

//! Decide whether the squared root is computed
#define IS_SQRT		0

//! Decide whether multiple points are computed in a single kernel call
#define IS_COMP_MULTI_POINTS	1
#if	IS_COMP_MULTI_POINTS
	#define MAX_NR_OF_COMP_POINTS	128
#endif	// #if	IS_COMP_MULTI_POINTS

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

void
_CompTransform
(
	const float4& f4A,
	const float4& f4B,
	const float4& f4C,
	Lib3dsVector v3X,
	Lib3dsVector v3Y,
	Lib3dsVector v3Z,
	Lib3dsVector v3B2,
	Lib3dsVector v3C2,
	float& fDet
 );

/*

$Log$

*/
