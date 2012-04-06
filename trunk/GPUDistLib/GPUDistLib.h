#pragma once

void
_GPUDistUseCpu
(
		bool bIsUsingCpu
		);

void
_GPUDistPrintTiming
(
		bool bIsPrintingTiming
		);

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

// ADD-BY-LEETEN 04/05/2012-BEGIN
//! Initialized GPUDist
void
_GPUDistInit();

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
// ADD-BY-LEETEN 04/05/2012-END

/*

$Log$

*/
