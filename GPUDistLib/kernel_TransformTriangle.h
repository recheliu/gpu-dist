#pragma once

#include "device_Vector.h"

//! A 2D texture for Point2
static texture<float4, cudaTextureType2D, cudaReadModeElementType> t2Df4Points;

__global__ 
void 
_TransformTriangle_kernel
(
	unsigned int	uNrOfTrianglesInBatch,
	unsigned int	uPointTexWidth,
	unsigned int	uPointTexHeight,

	ulong4	pu4TriangleVertices[],

	float4	pf4Xs[],
	float4	pf4Ys[],
	float4	pf4Zs[],
	float4	pf4B2s[],
	float4	pf4C2s[],
	float	pfDets[]
)
{
	unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int uTriangle = uBlock * blockDim.x + threadIdx.x;

	if( uTriangle < uNrOfTrianglesInBatch )
	{
		ulong4	u4TriangleVertices = pu4TriangleVertices[uTriangle];
		float4 f4A, f4B, f4C;
		f4A = tex2D(t2Df4Points, u4TriangleVertices.x % uPointTexWidth, u4TriangleVertices.x / uPointTexWidth);
		f4B = tex2D(t2Df4Points, u4TriangleVertices.y % uPointTexWidth, u4TriangleVertices.y / uPointTexWidth);
		f4C = tex2D(t2Df4Points, u4TriangleVertices.z % uPointTexWidth, u4TriangleVertices.z / uPointTexWidth);

		float4 f4BA = make_float4(
			f4B.x - f4A.x,
			f4B.y - f4A.y,
			f4B.z - f4A.z,
			0.0f);
		float4 f4Z = F4Normalize_device(f4BA);

		// use the normal N of ABC (ie. the cross product of CA and BA) as X axis
		float4 f4CA = make_float4(
			f4C.x - f4A.x,
			f4C.y - f4A.y,
			f4C.z - f4A.z,
			0.0f);
		float4 f4X = F4Cross_device(f4CA, f4BA);
		f4X = F4Normalize_device(f4X);

		// use the cross product of BA and N as the y axis
		float4 f4Y = F4Cross_device(f4BA, f4X);
		f4Y = F4Normalize_device(f4Y);
		
		// transform B and C to the new coordinate
		float4 f4B2 = make_float4(
			FDot_device(f4BA, f4X),
			FDot_device(f4BA, f4Y),
			FDot_device(f4BA, f4Z),
			0.0f);

		float4 f4C2 = make_float4(
			FDot_device(f4CA, f4X),
			FDot_device(f4CA, f4Y),
			FDot_device(f4CA, f4Z),
			0.0f);

		float fDet = f4B2.y * f4C2.z - f4B2.z * f4C2.y;	// yv3B2[1] * v3C2[2] - v3B2[2] * v3C2[1];

		pf4Xs[uTriangle] = f4X;
		pf4Ys[uTriangle] = f4Y;
		pf4Zs[uTriangle] = f4Z;
		pf4B2s[uTriangle] = f4B2;
		pf4C2s[uTriangle] = f4C2;
		pfDets[uTriangle] = fDet;
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
