#pragma once

#include "device_Vector.h"

__constant__ float4	f4A_const;
__constant__ float4	f4B_const;
__constant__ float4	f4C_const;
__constant__ float4	f4X_const;
__constant__ float4	f4Y_const;
__constant__ float4	f4Z_const;
__constant__ float4	f4B2_const;
__constant__ float4	f4C2_const;
__constant__ float	fDet_const;

/*
__device__
float
FDot_device(float4 f4V1, float4 f4V2)
{
	return f4V1.x * f4V2.x + f4V1.y * f4V2.y + f4V1.z * f4V2.z;
}
*/

__device__
void
_CompDistToEdge2D_device
(
	float4 f4P,
	float4 f4A,
	float4 f4B,

	float *pfDist
)
{
	float2 f2AP = make_float2(
		f4P.y - f4A.y,
		f4P.z - f4A.z);
	float fAP = f2AP.x * f2AP.x + f2AP.y * f2AP.y;
	#if	IS_SQRT
	fAP = sqrtf(fAP);
	#endif	// #if	IS_SQRT

	float2 f2AB = make_float2(
		f4B.y - f4A.y,
		f4B.z - f4A.z);
	float fAB = sqrtf(f2AB.x * f2AB.x + f2AB.y * f2AB.y);

	float fD = fAP;
	if( 0.0f < fAB )
	{
		float fAP2 = (f2AB.x * f2AP.x + f2AB.y * f2AP.y) / fAB;
		if( 0.0f > fAP2 )
			fD = fAP;
		else
		if( fAB < fAP2 )
		{
			float2 f2BP = make_float2(	
				f4P.y - f4B.y,
				f4P.z - f4B.z);
			float fBP = f2BP.x * f2BP.x + f2BP.y * f2BP.y;
			#if	IS_SQRT
			fBP = sqrtf(fBP);
			#endif	// #if	IS_SQRT
			fD = fBP;
		}
		else
			#if	IS_SQRT
			fD = sqrtf(fAP * fAP - fAP2 * fAP2);
			#else	// #if	IS_SQRT
			fD = fAP - fAP2 * fAP2;
			#endif	// #if	IS_SQRT
	}

	fD = max(fD, 0.0f);	
	*pfDist = fD;
}

__device__
void
_CompDistToEdge_device
(
	float4 f4P,
	float4 f4A,
	float4 f4B,

	float *pfDist
)
{
	float4 f4AP = make_float4(
		f4P.x - f4A.x,
		f4P.y - f4A.y,
		f4P.z - f4A.z,
		0.0f);
	float fAP = f4AP.x * f4AP.x + f4AP.y * f4AP.y + f4AP.z * f4AP.z;
	#if	IS_SQRT
	fAP = sqrtf(fAP);
	#endif	// #if	IS_SQRT

	float4 f4AB = make_float4(
		f4B.x - f4A.x,
		f4B.y - f4A.y,
		f4B.z - f4A.z,
		0.0f);
	float fAB = sqrtf(f4AB.x * f4AB.x + f4AB.y * f4AB.y + f4AB.z * f4AB.z);

	float fD = fAP;
	if( 0.0f < fAB )
	{
		float fAP2 = (f4AB.x * f4AP.x + f4AB.y * f4AP.y + f4AB.z * f4AP.z) / fAB;
		if( 0.0f > fAP2 )
			fD = fAP;
		else
		if( fAB < fAP2 )
		{
			float4 f4BP = make_float4(	
				f4P.x - f4B.x,
				f4P.y - f4B.y,
				f4P.z - f4B.z,
				0.0f);
			float fBP = f4BP.x * f4BP.x + f4BP.y * f4BP.y + f4BP.z * f4BP.z;
			#if	IS_SQRT
			fBP = sqrtf(fBP);
			#endif	// #if	IS_SQRT
			fD = fBP;
		}
		else
			#if	IS_SQRT
			fD = sqrtf(fAP * fAP - fAP2 * fAP2);
			#else	// #if	IS_SQRT
			fD = fAP - fAP2 * fAP2;
			#endif	// #if	IS_SQRT
	}

	fD = max(fD, 0.0f);	
	*pfDist = fD;
}

__global__ 
void 
_CompDistFromTriangle_kernel
(
	unsigned int	uNrOfNeededThreads,
	float4		pf4Points1_device[],
	unsigned int	uTriangle,
	float		pfDists_device[]
)
{
	unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int uPoint1 = uBlock * blockDim.x + threadIdx.x;

	if( uPoint1 < uNrOfNeededThreads )
	{
		float4 f4P = pf4Points1_device[uPoint1];

		float fDist = 0.0f;
		if( !fDet_const )
		{
			float fD1, fD2, fD3;
			_CompDistToEdge_device(f4P,	f4B_const,	f4C_const,	&fD1);	fDist = fD1;
			_CompDistToEdge_device(f4P,	f4A_const,	f4B_const,	&fD2);	fDist = min(fDist, fD2);
			_CompDistToEdge_device(f4P,	f4A_const,	f4C_const,	&fD3);	fDist = min(fDist, fD3);
		}
		else
		{

		// and transform P to the new coordinate 
		float4 f4PA = make_float4(
			f4P.x - f4A_const.x,
			f4P.y - f4A_const.y,
			f4P.z - f4A_const.z,
			1.0f);

		float4 f4P2 = make_float4(
			FDot_device(f4PA, f4X_const),
			FDot_device(f4PA, f4Y_const),
			FDot_device(f4PA, f4Z_const),
			1.0f);

		// solve (fS, fT) s.t. v3P2 = fS * v3C2 + fT * v3BA
		// v3P2[1]		v3B2[1]	v3C2[1]		fS
		//		= [			] [		]
		// v3P2[2]		v3B2[2]	v3C2[2]		fT
		// ->
		//		+v3C2[2]	-v3C2[1]	v3P2[1]		fS
		// ___	[				] [		] = [		]
		// det		-v3B2[2]	+v3B2[1]	v3P2[2]		fT
		float fS = (+f4C2_const.z * f4P2.y - f4C2_const.y * f4P2.z) / fDet_const;
		float fT = (-f4B2_const.z * f4P2.y + f4B2_const.y * f4P2.z) / fDet_const;

		fDist = fabsf(f4P2.x);
		#if	!IS_SQRT	
		fDist *= fDist;
		#endif	// #if	!IS_SQRT	
		if( !(
			0.0f < fS && fS < 1.0f &&
			0.0f < fT && fT < 1.0f &&
			fS + fT < 1.0f	) )
		{
			float4 f4Zero = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			float fD1, fD2, fD3;
			float fDistToEdge2D;
			_CompDistToEdge2D_device(f4P2,	f4B2_const,	f4C2_const,	&fD1);	fDistToEdge2D = fD1;
			_CompDistToEdge2D_device(f4P2,	f4Zero,		f4B2_const,	&fD2);	fDistToEdge2D = min(fDistToEdge2D, fD2);
			_CompDistToEdge2D_device(f4P2,	f4Zero,		f4C2_const,	&fD3);	fDistToEdge2D = min(fDistToEdge2D, fD3);
			#if	!IS_SQRT	
			fDist = fDistToEdge2D + fDist;
			#else	// #if	!IS_SQRT	
			fDist = sqrtf(fDistToEdge2D * fDistToEdge2D + fDist * fDist);
			#endif	// #if	!IS_SQRT	
		}
		}	

		if( uTriangle > 0 )
		{
			float fOldDist = pfDists_device[uPoint1];
			fDist = min(fDist, fOldDist);
		}
		pfDists_device[uPoint1] = fDist;
	}
}

__constant__ float4 f4Dir_const;
__constant__ float fThreshold_const = 0.00001f;

__global__ 
void 
_CountIntersectingTriangle_kernel
(
	unsigned int	uNrOfNeededThreads,
	float4		pf4Points_device[],
	float		pfCounts_device[]
)
{
	unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int uPoint = uBlock * blockDim.x + threadIdx.x;

	if( uPoint < uNrOfNeededThreads )
	{
		float4 f4P = pf4Points_device[uPoint];
		float4 f4Dir = f4Dir_const;

		// transform P to the new coordinate 
		float4 f4PA = make_float4(
			f4P.x - f4A_const.x,
			f4P.y - f4A_const.y,
			f4P.z - f4A_const.z,
			1.0f);

		float4 f4P2 = make_float4(
			FDot_device(f4PA, f4X_const),
			FDot_device(f4PA, f4Y_const),
			FDot_device(f4PA, f4Z_const),
			1.0f);

		float fY2, fZ2;
		bool bIsIntersectionImpossible = false;
		if( fThreshold_const > fabsf(f4P2.x) )
		{
			fY2 = f4P2.y;
			fZ2 = f4P2.z;
			bIsIntersectionImpossible = true;
		}
		else
		{
			// compute a point q along the direction (1, 0, 0);
			float4 f4Q = make_float4(
				f4P.x + f4Dir.x,
				f4P.y + f4Dir.y,
				f4P.z + f4Dir.z,
				f4P.w + f4Dir.w);			

			// transform Q to the new coordinate 
			float4 f4QA = make_float4(
				f4Q.x - f4A_const.x,
				f4Q.y - f4A_const.y,
				f4Q.z - f4A_const.z,
				1.0f);

			float4 f4Q2 = make_float4(
				FDot_device(f4QA, f4X_const),
				FDot_device(f4QA, f4Y_const),
				FDot_device(f4QA, f4Z_const),
				1.0f);

			float fScale = f4Q2.x - f4P2.x;
			if( 0.0f != fScale )
			{
				float fStep = (0.0f - f4P2.x) / fScale;
				if( fStep > 0 )
				{
					fY2 = f4P2.y + (f4Q2.y - f4P2.y) * fStep;
					fZ2 = f4P2.z + (f4Q2.z - f4P2.z) * fStep;
				}
				else
					bIsIntersectionImpossible = true;
			}
			else
				bIsIntersectionImpossible = true;
		}

		// compute the intesection between the vector from Q2 to P2 and the plan X = 0
		float fNewCount = 0.0f;
		if( !bIsIntersectionImpossible ) 
		{
			// solve (fS, fT) s.t. v3P2 = fS * v3C2 + fT * v3BA
			// fY2		v3B2[1]	v3C2[1]		fS
			//	= [			] [		]
			// fZ2		v3B2[2]	v3C2[2]		fT
			// ->
			//		+v3C2[2]	-v3C2[1]	fY2		fS
			// ___	[				] [		] = [		]
			// det		-v3B2[2]	+v3B2[1]	fZ2		fT
			float fS = (+f4C2_const.z * fY2 - f4C2_const.y * fZ2) / fDet_const;
			float fT = (-f4B2_const.z * fY2 + f4B2_const.y * fZ2) / fDet_const;

			if(	fThreshold_const < fabsf(f4P2.x) &&
				0.0f + fThreshold_const <= fS		&& fS		<= 1.0f - fThreshold_const &&
				0.0f + fThreshold_const <= fT		&& fT		<= 1.0f - fThreshold_const &&
				0.0f + fThreshold_const <= fS + fT	&& fS + fT	<= 1.0f	- fThreshold_const )
			{
				fNewCount = 1.0f;
			}
		}
		pfCounts_device[uPoint] += fNewCount;
	}
}


/*

$Log: kernel_CompComplexity.h,v $
Revision 1.1  2012-03-30 06:00:33  leeten

[03/29/2012]
<POMCuda_host.cu>
<POMCuda.h>
1. [ADD] Include the header assert.h.
2. [ADD] Define functions _PomCompComplexityFree(), _PomCompComplexityInit(), and _PomCompComplexity() to compute the complexity of all POMs on GPUs.

<CMakeLists.txt>
<kernel_CompComplexity.h>
1. [1ST] Check in the kernel to compute the complexity.


*/
