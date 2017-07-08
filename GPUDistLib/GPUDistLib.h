#pragma once

//! Return a boolean whether the distance is the squared of the true disance. 
/*!
If the return value is false, the caller should compute the squared too for the distance.
*/
bool
BGPUDistIsDistSquaredRoot
(
		);

//! Specified whether CPUs (true) or GPUs (false) is used
void
_GPUDistUseCpu
(
		bool bIsUsingCpu
		);

//! Specified whether the timing is printed.
void
_GPUDistPrintTiming
(
		bool bIsPrintingTiming
		);

//! Compute the distance from each grid point to the input points
void
_GPUDistComputeDistanceFieldFromPoints
(
	size_t uNrOfPoints,
	float4 pf4Points[],
	size_t uWidth,
	size_t uHeight,
	size_t uDepth,
	float pfDist[]
);

//! Compute the distance from each grid point to the input points on CPUs
void
_GPUDistComputeDistanceFieldFromPointsByCpu
(
	size_t uNrOfPoints,
	float4 pf4Points[],
	size_t uWidth,
	size_t uHeight,
	size_t uDepth,
	float pfDist[]
);

//! Compute the distance from each point in Points1 to the sqeuence of points (Points2)
void
_GPUDistCompDistFromPointsToPoints
(
	size_t uNrOfPoints1,
	const float4 pf4Points1[],

	size_t uNrOfPoints2,
	const float4 pf4Points2[],

	float pfDist[],
	unsigned int *puNearestPoint2 = NULL,
	void *pReserved = NULL
);

//! Compute the distance from each point in Points1 to the sqeuence of points (Points2) on CPUs.
void
_GPUDistCompDistFromPointsToPointsByCpu
(
	size_t uNrOfPoints1,
	const float4 pf4Points1[],

	size_t uNrOfPoints2,
	const float4 pf4Points2[],

	float pfDist[],
	unsigned int *puNearestPoint2 = NULL,
	void *pReserved = NULL
);

//! Compute distance from each point in Points1 to the triangle meshes defined by the points (Points2) and the connectivity on CPUs.
void 
_GPUDistCompDistFromPointsToTrianglesByCpu
(
	size_t uNrOfPoints1,
	const float4 pf4Points1[],

	size_t uNrOfPoints2,
	const float4 pf4Points2[],

	size_t uNrOfTriangles,
	const ulong4 pu4TriangleVertices[],

	bool bIsPrecomputingTrasforms,

	float pfDists[]
);

void 
//! Compute distance from each point in Points1 to the triangle meshes defined by the points (Points2) and the connectivity.
_GPUDistCompDistFromPointsToTriangles
(
	size_t uNrOfPoints1,
	const float4 pf4Points1[],

	size_t uNrOfPoints2,
	const float4 pf4Points2[],

	size_t uNrOfTriangles,
	const ulong4 pu4TriangleVertices[],

	bool bIsPrecomputingTrasforms,

	float pfDists[]
);

void
_GPUDistCountIntersectingTriangles
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	size_t uNrOfTriangles,
	ulong4 pu4TriangleVertices[],

	float4 f4Dir,

	float pfCount[]
);

//! Initialized GPUDist
void
_GPUDistInit();

#if	defined(WITH_CUDPP)	
//////////////// Functions to compute Hausdorff distance /////////////////////////////

//! Setup the length of the two sequecnes to be compared
/*!
This function should be called before _GPUHausSetPoints() and _GPUHausComp().
*/
void
_GPUHausSetLengths
(
	size_t uNrOfPoints1,
	size_t uNrOfPoints2
);

//! Setup the sequecne to be compared
void
_GPUHausSetPoints
(
	int iIndex,
	size_t uNrOfPoints,
	float4 pf4Points[]
);

//! Compute the Hausdorff distance between the two sequecne
void
_GPUHausComp
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	float* pfDist1To2
);

void
_GPUHausCompByCpu
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	float* pfDist1To2
);

void
_cuDistInit
(
	size_t uBatchSize
);

void
_cuDistSetLengths
(
	size_t uNrOfPoints1,
	size_t uNrOfPoints2
);

void
_cuDistComp
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	float pfDists[]
);

#endif	// #if	defined(WITH_CUDPP)	

/*

$Log$

*/
