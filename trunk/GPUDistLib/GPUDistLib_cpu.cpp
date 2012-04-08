#include "GPUDistLib_internal.h"

void
_GPUDistComputeDistanceFieldFromPointsByCpu
(
	size_t uNrOfPoints,
	float4 pf4Points[],
	size_t uWidth,
	size_t uHeight,
	size_t uDepth,
	float pfDist[]
)
{
LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);
LIBCLOCK_BEGIN(bIsPrintingTiming);
	for(size_t v = 0,	d = 0; d < uDepth; d++)
		for(size_t		h = 0; h < uHeight; h++)
			for(size_t	w = 0; w < uWidth; w++, v++)
			{
				float fDist = (float)HUGE_VAL;
				for(size_t p = 0; p < uNrOfPoints; p++)
				{
					float fDx = (float)w - pf4Points[p].x;
					float fDy = (float)h - pf4Points[p].y;
					float fDz = (float)d - pf4Points[p].z;
					float fD = fDx * fDx + fDy * fDy + fDz * fDz;
					fDist = min(fDist, fD);
				}
#if	IS_SQRT
				fDist = sqrtf(fDist);
#endif
				pfDist[v] = fDist;
			}
LIBCLOCK_END(bIsPrintingTiming);
LIBCLOCK_PRINT(bIsPrintingTiming);
}


void
_GPUDistCompDistFromPointsToPointsByCpu
(
	size_t uNrOfPoints1,
	float4 pf4Points1[],

	size_t uNrOfPoints2,
	float4 pf4Points2[],

	float pfDists[]
)
{
	LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);
	LIBCLOCK_BEGIN(bIsPrintingTiming);
	for(size_t p1 = 0; p1 < uNrOfPoints1; p1++)
	{
		float fMinDist = (float)HUGE_VAL;
		for(size_t p2 = 0; p2 < uNrOfPoints2; p2++)
		{
			float fDx = pf4Points2[p2].x - pf4Points1[p1].x;
			float fDy = pf4Points2[p2].y - pf4Points1[p1].y;
			float fDz = pf4Points2[p2].z - pf4Points1[p1].z;
			float fDist = fDx * fDx + fDy * fDy + fDz * fDz;
			fMinDist = min(fMinDist, fDist);
		}
#if	IS_SQRT
		fMinDist = sqrtf(fMinDist);
#endif
		pfDists[p1] = fMinDist;
	}
	LIBCLOCK_END(bIsPrintingTiming);
	LIBCLOCK_PRINT(bIsPrintingTiming);
}


/*

$Log$

*/