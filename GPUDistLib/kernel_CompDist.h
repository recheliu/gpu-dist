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
	float4 	f4Point,
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
		float fDx = fX - f4Point.x;
		float fDy = fY - f4Point.y;
		float fDz = fZ - f4Point.z;
		float fDist = sqrt(fDx * fDx + fDy * fDy + fDz * fDz);

		if( iPoint > 0 )
			fDist = min(fDist, fOldDist);

		pfDists_device[uI] = fDist;
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
