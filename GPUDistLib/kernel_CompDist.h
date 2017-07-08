__global__ 
void 
_CompDist_kernel
(
	unsigned int uThreadOffset,
	int iPoint,
	#if	IS_COMP_MULTI_POINTS
	unsigned int uNrOfPointsToCompare,
	#else	// #if	IS_COMP_MULTI_POINTS
	float4 	f4Point,
	#endif	// #if	IS_COMP_MULTI_POINTS	
	unsigned int uWidth,
	unsigned int uHeight,
	unsigned int uDepth,
	float pfDists_device[]
)
{
	unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int uI = uBlock * blockDim.x + threadIdx.x;
	unsigned int uV = uThreadOffset + uI;
	unsigned int uNrOfVoxels = uWidth * uHeight * uDepth;

	if( uV < uNrOfVoxels )
	{
		float fOldDist = pfDists_device[uI];
		float fX = (float)( uV % uWidth );
		float fY = (float)( (uV / uWidth) % uHeight);
		float fZ = (float)( uV / (uWidth * uHeight) );
		#if	IS_COMP_MULTI_POINTS
		float fD;
		for(unsigned int pi = 0; pi < uNrOfPointsToCompare; pi++)
		{
			float4 f4Point = pf4Points_const[pi];
		#endif	// #if	IS_COMP_MULTI_POINTS

		float fDx = fX - f4Point.x;
		float fDy = fY - f4Point.y;
		float fDz = fZ - f4Point.z;
		float fDist = fDx * fDx + fDy * fDy + fDz * fDz;
		#if	IS_SQRT
		fDist = sqrtf(fDist);
		#endif	// #if	IS_SQRT

		#if	IS_COMP_MULTI_POINTS
		fD = (pi > 0)?min(fD, fDist):fDist;
		}
		float fDist = fD;
		#endif	// #if	IS_COMP_MULTI_POINTS

		if( iPoint > 0 )
			fDist = min(fDist, fOldDist);

		pfDists_device[uI] = fDist;
	}
}

__global__ 
void 
_CompDistFromPoints_kernel
(
	unsigned int uNrOfPoints1,
	float4 pf4Points1_device[],

	unsigned int uPoint2,
	#if	IS_COMP_MULTI_POINTS			
	unsigned int uNrOfPoints2ToCompare,
	#else	// #if	IS_COMP_MULTI_POINTS			
	float4 	f4Point2,
	#endif	//	#if	!IS_COMP_MULTI_POINTS	

	float pfDists_device[],

	unsigned int puNearestPoint2_device[],

	void* pReserved
)
{
	unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int uPoint1 = uBlock * blockDim.x + threadIdx.x;

	if( uPoint1 < uNrOfPoints1 )
	{
		float4 f4Point1 = pf4Points1_device[uPoint1];
		unsigned int uNearestPoint = 0; // the index of the smallest distance in this iteration
		#if	IS_COMP_MULTI_POINTS
		float fD;
		for(unsigned int p2i = 0; p2i < uNrOfPoints2ToCompare; p2i++)
		{
			float4 f4Point2 = pf4Points_const[p2i];
		#endif	// #if	IS_COMP_MULTI_POINTS

		float fDx = f4Point1.x - f4Point2.x;
		float fDy = f4Point1.y - f4Point2.y;
		float fDz = f4Point1.z - f4Point2.z;
		float fDist = fDx * fDx + fDy * fDy + fDz * fDz;

		#if	IS_SQRT
		fDist = sqrtf(fDist);
		#endif	// #if	IS_SQRT

		#if	IS_COMP_MULTI_POINTS
		if(0 == p2i)
		{
			uNearestPoint = 0;
			fD = fDist;
		}
		else
		if( fD > fDist )
		{
			uNearestPoint = p2i;
			fD = fDist;
		}
		}
		float fDist = fD;
		#endif	// #if	IS_COMP_MULTI_POINTS
		unsigned int uNearestPoint2 = 0;
		if( NULL != puNearestPoint2_device && uPoint2 > 0 )
			uNearestPoint2 = puNearestPoint2_device[uPoint1];
		if( uPoint2 > 0 )
		{
			float fOldDist = pfDists_device[uPoint1];
			if( fDist < fOldDist )
				uNearestPoint2 = uPoint2 + uNearestPoint;
			else
				fDist = fOldDist;
		}

		pfDists_device[uPoint1] = fDist;
		if( NULL != puNearestPoint2_device )
			puNearestPoint2_device[uPoint1] = uNearestPoint2;
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
