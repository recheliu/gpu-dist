//! A 1D texture for current computed distance
static texture<float, 1, cudaReadModeElementType> t1DfDistsInBatch;

#if	POINT1_AS	==	POINT_AS_CONSTANT
static __constant__ float4 pf4Points1InBatch_const[NR_OF_POINTS1_PER_BATCH];
#else
//! A 1D texture for Point1
static texture<float4, cudaTextureType1D, cudaReadModeElementType> t1Df4Points1InBatch;
#endif

//! A 2D texture for Point2
static texture<float4, cudaTextureType2D, cudaReadModeElementType> t2Df4Points2InBatch;

//
static
__global__ 
void 
_CompPairDistInBatch_kernel
(
	unsigned int uBatch2,
	unsigned int uNrOfPoints1InBatch,
	unsigned int uNrOfPoints2InBatch,
	unsigned int uPoint2TexWidth,
	float pfDists_device[]
)
{
	unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int uThread = uBlock * blockDim.x + threadIdx.x;
	unsigned int uNrOfPairs = uNrOfPoints1InBatch * uNrOfPoints2InBatch;

	if( uThread < uNrOfPairs )
	{
		unsigned int uPoint2 = uThread % uNrOfPoints2InBatch;
		unsigned int uPoint2X = uPoint2 % uPoint2TexWidth;
		unsigned int uPoint2Y = uPoint2 / uPoint2TexWidth;
		float4 f4Point2 = tex2D(t2Df4Points2InBatch, uPoint2X, uPoint2Y);

		unsigned int uPoint1 = uThread / uNrOfPoints2InBatch;
		float4 f4Point1;
		#if	POINT1_AS	==	POINT_AS_CONSTANT
		f4Point1 = pf4Points1InBatch_const[uPoint1];
		#elif	POINT1_AS	==	POINT_AS_LINEAR_BUFFER
		f4Point1 = tex1Dfetch(t1Df4Points1InBatch, uPoint1);
		#elif	POINT1_AS	==	POINT_AS_ARRAY
		f4Point1 = tex1D(t1Df4Points1InBatch, (float)uPoint1);
		#endif	// #if POINT_AS

		float fDx = f4Point1.x - f4Point2.x;
		float fDy = f4Point1.y - f4Point2.y;
		float fDz = f4Point1.z - f4Point2.z;
		float fDist = fDx * fDx + fDy * fDy + fDz * fDz;

		if( uBatch2 > 0 )
		{
			float fD = tex1Dfetch(t1DfDistsInBatch, uPoint1);
			fDist = min(fD, fDist);
		}

		pfDists_device[uThread] = fDist;
	}
}

/*

$Log$

*/
