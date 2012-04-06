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

	int iNrOfTrials = 100;
	for(int le = 5; le <= 12; le++)
	{
		int iLength = 1 << le;	// 4096;

		TBuffer<float4> pf4Points1;
		pf4Points1.alloc(iLength);
		for(size_t p = 0; p < pf4Points1.USize(); p++)
			pf4Points1[p] = make_float4(
				(float)(rand() % pf4Points1.USize()) / (float)pf4Points1.USize(),
				(float)(rand() % pf4Points1.USize()) / (float)pf4Points1.USize(),
				(float)(rand() % pf4Points1.USize()) / (float)pf4Points1.USize(),
				1.0f);
		LOG_VAR(pf4Points1.USize());

		TBuffer<float4> pf4Points2;
		pf4Points2.alloc(iLength);
		for(size_t p = 0; p < pf4Points2.USize(); p++)
			pf4Points2[p] = make_float4(
				(float)(rand() % pf4Points2.USize()) / (float)pf4Points2.USize(),
				(float)(rand() % pf4Points2.USize()) / (float)pf4Points2.USize(),
				(float)(rand() % pf4Points2.USize()) / (float)pf4Points2.USize(),
				1.0f);
		LOG_VAR(pf4Points2.USize());

		float fDist1To2;

		LIBCLOCK_INIT(1, __FUNCTION__);
		LIBCLOCK_BEGIN(1);

		// first, setup the sizes of both sets of points
		for(int i = 0; i < iNrOfTrials; i++)
		{
			_GPUHausSetLengths
			(
				pf4Points1.USize(),
				pf4Points2.USize()
			);

			// specify the points of the first set
			_GPUHausSetPoints
			(
				0,
				pf4Points1.USize(),
				&pf4Points1[0]
			);
			// specify the points of the 2nd set
			_GPUHausSetPoints
			(
				1,
				pf4Points2.USize(),
				&pf4Points2[0]
			);
		}
		LIBCLOCK_END(1);

		LIBCLOCK_BEGIN(1);
		// CPU
		_GPUDistUseCpu(false);
		for(int i = 0; i < iNrOfTrials; i++)

			// compute the Hausdorff distance between the two specified sets of points
			_GPUHausComp
			(
				pf4Points1.USize(),
				&pf4Points1[0],

				pf4Points2.USize(),
				&pf4Points2[0],

				&fDist1To2
			);
		LIBCLOCK_END(1);
		LOG_VAR(fDist1To2);

		LIBCLOCK_BEGIN(1);

		// now use cpu for reference
		_GPUDistUseCpu(true);	// GPU
		for(int i = 0; i < iNrOfTrials; i++)
			_GPUHausComp
			(
				pf4Points1.USize(),
				&pf4Points1[0],

				pf4Points2.USize(),
				&pf4Points2[0],

				&fDist1To2
			);
		LIBCLOCK_END(1);
		LOG_VAR(fDist1To2);

		LIBCLOCK_PRINT(1);
	}

	return 0;
}