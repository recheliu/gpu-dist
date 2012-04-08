#include <math.h>
#include <vector_functions.h>

#include "liblog.h"
#include "libclock.h"
#include "libbuf.h"
#include "libbuf3d.h"

#include "GPUDistLib.h"

int 
main(int argn, char* argv[])
{
	// init. GPUDist
	_GPUDistInit();

	// setup GPUDistLib
	_GPUDistPrintTiming(false);

	int iNrOfTrials = 1;
	for(int le = 15; le <= 15; le++)
	{
		int iLength = 1 << le;	// 4096;

		TBuffer<float4> pf4Points1;
		pf4Points1.alloc(4000000);	// iLength);
		for(size_t p = 0; p < pf4Points1.USize(); p++)
			pf4Points1[p] = make_float4(
				(float)(rand() % pf4Points1.USize()) / (float)pf4Points1.USize(),
				(float)(rand() % pf4Points1.USize()) / (float)pf4Points1.USize(),
				(float)(rand() % pf4Points1.USize()) / (float)pf4Points1.USize(),
				1.0f);
		LOG_VAR(pf4Points1.USize());

		TBuffer<float4> pf4Points2;
		pf4Points2.alloc(128000);	// iLength);
		for(size_t p = 0; p < pf4Points2.USize(); p++)
			pf4Points2[p] = make_float4(
				(float)(rand() % pf4Points2.USize()) / (float)pf4Points2.USize(),
				(float)(rand() % pf4Points2.USize()) / (float)pf4Points2.USize(),
				(float)(rand() % pf4Points2.USize()) / (float)pf4Points2.USize(),
				1.0f);
		LOG_VAR(pf4Points2.USize());

		TBuffer<float> pfDist;
		pfDist.alloc(pf4Points1.USize());

		LIBCLOCK_INIT(1, __FUNCTION__);

		// first, setup the sizes of both sets of points
		LIBCLOCK_BEGIN(1);
		_cuDistSetLengths
		(
			pf4Points1.USize(),
			pf4Points2.USize()
		);
		LIBCLOCK_END(1);

		// compute the Hausdorff distance between the two specified sets of points
		LIBCLOCK_BEGIN(1);
		for(int ti = 0; ti < iNrOfTrials; ti++)
		{
			_cuDistComp
			(
				pf4Points1.USize(),
				&pf4Points1[0],

				pf4Points2.USize(),
				&pf4Points2[0],

				&pfDist[0]
			);
		}
		LIBCLOCK_END(1);

		LIBCLOCK_BEGIN(1);
		float fDist1To2 = -(float)HUGE_VAL;
		for(int i = 0; i < pfDist.USize(); i++)
			fDist1To2 = max(fDist1To2, pfDist[i]);
		fDist1To2 = sqrtf(fDist1To2);
		LOG_VAR(fDist1To2);
		LIBCLOCK_END(1);
		LIBCLOCK_PRINT(1);

		LIBCLOCK_INIT(1, __FUNCTION__);
		LIBCLOCK_BEGIN(1);
		float fDist1To2;
		for(int ti = 0; ti < iNrOfTrials; ti++)
		{
			_GPUHausCompByCpu
			(
				pf4Points1.USize(),
				&pf4Points1[0],

				pf4Points2.USize(),
				&pf4Points2[0],

				&fDist1To2
			);
		}
		LOG_VAR(fDist1To2);
		LIBCLOCK_END(1);
		LIBCLOCK_PRINT(1);
	}


	return 0;
}
