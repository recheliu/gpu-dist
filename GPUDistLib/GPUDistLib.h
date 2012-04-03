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

/*

$Log$

*/
