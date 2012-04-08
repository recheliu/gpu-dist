__global__ 
void 
_CompDist_kernel
(
	#if	0	// MOD-BY-LEETEN 04/04/2012-FROM:
		size_t uThreadOffset,
		int iPoint,
		float4 	f4Point,
		size_t	uWidth,
		size_t	uHeight,
		size_t	uDepth,
		float pfDists_device[]
	#else		// MOD-BY-LEETEN 04/04/2012-TO:
	unsigned int uThreadOffset,
	int iPoint,
	// ADD-BY-LEETEN 04/07/2012-BEGIN
	#if	IS_COMP_MULTI_POINTS
	unsigned int uNrOfPointsToCompare,
	#else	// #if	IS_COMP_MULTI_POINTS
	// ADD-BY-LEETEN 04/07/2012-END
	float4 	f4Point,
	#endif	// #if	IS_COMP_MULTI_POINTS	// ADD-BY-LEETEN 04/07/2012
	unsigned int uWidth,
	unsigned int uHeight,
	unsigned int uDepth,
	float pfDists_device[]
	#endif		// MOD-BY-LEETEN 04/04/2012-END
)
{
	#if	0	// MOD-BY-LEETEN 04/04/2012-FROM:
		size_t uBlock = gridDim.x * blockIdx.y + blockIdx.x;
		size_t uI = uBlock * blockDim.x + threadIdx.x;
		size_t uV = uThreadOffset + uI;
		size_t uNrOfVoxels = uWidth * uHeight * uDepth;
	#else		// MOD-BY-LEETEN 04/04/2012-TO:
	unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int uI = uBlock * blockDim.x + threadIdx.x;
	unsigned int uV = uThreadOffset + uI;
	unsigned int uNrOfVoxels = uWidth * uHeight * uDepth;
	#endif		// MOD-BY-LEETEN 04/04/2012-END

	if( uV < uNrOfVoxels )
	{
		float fOldDist = pfDists_device[uI];
		float fX = (float)( uV % uWidth );
		float fY = (float)( (uV / uWidth) % uHeight);
		float fZ = (float)( uV / (uWidth * uHeight) );
		// ADD-BY-LEETEN 04/07/2012-BEGIN
		#if	IS_COMP_MULTI_POINTS
		float fD;
		for(unsigned int pi = 0; pi < uNrOfPointsToCompare; pi++)
		{
			float4 f4Point = pf4Points_const[pi];
		#endif	// #if	IS_COMP_MULTI_POINTS
		// ADD-BY-LEETEN 04/07/2012-END

		float fDx = fX - f4Point.x;
		float fDy = fY - f4Point.y;
		float fDz = fZ - f4Point.z;
		// MOD-BY-LEETEN 04/07/2012-FROM:		float fDist = sqrt(fDx * fDx + fDy * fDy + fDz * fDz);
		float fDist = fDx * fDx + fDy * fDy + fDz * fDz;
		// MOD-BY-LEETEN 04/07/2012-END
		// ADD-BY-LEETEN 04/07/2012-BEGIN
		#if	IS_SQRT
		fDist = sqrtf(fDist);
		#endif	// #if	IS_SQRT
		// ADD-BY-LEETEN 04/07/2012-END

		// ADD-BY-LEETEN 04/07/2012-BEGIN
		#if	IS_COMP_MULTI_POINTS
		fD = (pi > 0)?min(fD, fDist):fDist;
		}
		float fDist = fD;
		#endif	// #if	IS_COMP_MULTI_POINTS
		// ADD-BY-LEETEN 04/07/2012-END

		if( iPoint > 0 )
			fDist = min(fDist, fOldDist);

		pfDists_device[uI] = fDist;
	}
}

// ADD-BY-LEETEN 04/07/2012-BEGIN
__global__ 
void 
_CompDistFromPoints_kernel
(
	unsigned int uNrOfPoints1,
	float4 pf4Points1_device[],

	unsigned int uPoint2,
	// ADD-BY-LEETEN 04/07/2012-BEGIN
	#if	IS_COMP_MULTI_POINTS			
	unsigned int uNrOfPoints2ToCompare,
	#else	// #if	IS_COMP_MULTI_POINTS			
	// ADD-BY-LEETEN 04/07/2012-END
	float4 	f4Point2,
	#endif	//	#if	!IS_COMP_MULTI_POINTS	// ADD-BY-LEETEN 04/07/2012

	float pfDists_device[]
)
{
	unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int uPoint1 = uBlock * blockDim.x + threadIdx.x;

	if( uPoint1 < uNrOfPoints1 )
	{
		float4 f4Point1 = pf4Points1_device[uPoint1];
		// ADD-BY-LEETEN 04/07/2012-BEGIN
		#if	IS_COMP_MULTI_POINTS
		float fD;
		for(unsigned int p2i = 0; p2i < uNrOfPoints2ToCompare; p2i++)
		{
			float4 f4Point2 = pf4Points_const[p2i];
		#endif	// #if	IS_COMP_MULTI_POINTS
		// ADD-BY-LEETEN 04/07/2012-END

		float fDx = f4Point1.x - f4Point2.x;
		float fDy = f4Point1.y - f4Point2.y;
		float fDz = f4Point1.z - f4Point2.z;
		float fDist = fDx * fDx + fDy * fDy + fDz * fDz;

		// ADD-BY-LEETEN 04/07/2012-BEGIN
		#if	IS_SQRT
		fDist = sqrtf(fDist);
		#endif	// #if	IS_SQRT
		// ADD-BY-LEETEN 04/07/2012-END

		// ADD-BY-LEETEN 04/07/2012-BEGIN
		#if	IS_COMP_MULTI_POINTS
			fD = (p2i > 0)?min(fD, fDist):fDist;
		}
		float fDist = fD;
		#endif	// #if	IS_COMP_MULTI_POINTS
		// ADD-BY-LEETEN 04/07/2012-END

		if( uPoint2 > 0 )
		{
			float fOldDist = pfDists_device[uPoint1];
			fDist = min(fDist, fOldDist);
		}

		pfDists_device[uPoint1] = fDist;
	}
}
// ADD-BY-LEETEN 04/07/2012-END

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
