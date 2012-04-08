#if	0	// MOD-BY-LEETEN 04/07/2012-FROM:
	//! A 1D texture for Point1
	texture<float4, 1, cudaReadModeElementType> t1Df4Points1;

	//! A 1D texture for Point2
	texture<float4, 1, cudaReadModeElementType> t1Df4Points2;
#else	// MOD-BY-LEETEN 04/07/2012-TO:
//! A 1D texture for Point1
static texture<float4, 1, cudaReadModeElementType> t1Df4Points1;

//! A 1D texture for Point2
static texture<float4, 1, cudaReadModeElementType> t1Df4Points2;
#endif	// MOD-BY-LEETEN 04/07/2012-END

//
__global__ 
static
void 
_CompPairDist_kernel
(
	unsigned int uNrOfPoints1,
	unsigned int uNrOfPoints2,
	float pfDists_device[]
)
{
	unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int uThread = uBlock * blockDim.x + threadIdx.x;
	unsigned int uNrOfPairs = uNrOfPoints1 * uNrOfPoints2;

	if( uThread < uNrOfPairs )
	{
		unsigned int uPoint2 = uThread % uNrOfPoints2;
		unsigned int uPoint1 = uThread / uNrOfPoints2;
		float4 f4Point1 = tex1Dfetch(t1Df4Points1, uPoint1);
		float4 f4Point2 = tex1Dfetch(t1Df4Points2, uPoint2);
		float fDx = f4Point1.x - f4Point2.x;
		float fDy = f4Point1.y - f4Point2.y;
		float fDz = f4Point1.z - f4Point2.z;
		float fDist = fDx * fDx + fDy * fDy + fDz * fDz;

		pfDists_device[uThread] = fDist;
	}
}

#if	0	// DEL-BY-LEETEN 04/07/2012-BEGIN
	__global__ 
	void 
	_SetupSegments_kernel
	(
		unsigned int uNrOfPoints1,
		unsigned int uNrOfPoints2,
		unsigned int puSegBegin_device[]
	)
	{
		unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
		unsigned int uThread = uBlock * blockDim.x + threadIdx.x;

		if( uThread < uNrOfPoints1 )
			puSegBegin_device[uThread * uNrOfPoints2] = 1;
	}
#endif	// DEL-BY-LEETEN 04/07/2012-END

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
