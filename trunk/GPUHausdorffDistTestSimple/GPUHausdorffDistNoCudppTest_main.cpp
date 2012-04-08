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
	_GPUDistPrintTiming(true);

	// MOD-BY-LEETEN 04/07/2012-FROM:	int iNrOfTrials = 100;
	int iNrOfTrials = 1;
	// MOD-BY-LEETEN 04/07/2012-END
	for(int le = 18; le <= 18; le++)
	{
		int iLength = 1 << le;	// 4096;

		TBuffer<float4> pf4Points1;
		pf4Points1.alloc(400000);	// iLength);
		for(size_t p = 0; p < pf4Points1.USize(); p++)
			pf4Points1[p] = make_float4(
				(float)(rand() % pf4Points1.USize()) / (float)pf4Points1.USize(),
				(float)(rand() % pf4Points1.USize()) / (float)pf4Points1.USize(),
				(float)(rand() % pf4Points1.USize()) / (float)pf4Points1.USize(),
				1.0f);
		LOG_VAR(pf4Points1.USize());

		TBuffer<float4> pf4Points2;
		pf4Points2.alloc( 12800 );	// (int)powf((float)iLength, 0.66667f) );
		for(size_t p = 0; p < pf4Points2.USize(); p++)
			pf4Points2[p] = make_float4(
				(float)(rand() % pf4Points2.USize()) / (float)pf4Points2.USize(),
				(float)(rand() % pf4Points2.USize()) / (float)pf4Points2.USize(),
				(float)(rand() % pf4Points2.USize()) / (float)pf4Points2.USize(),
				1.0f);
		LOG_VAR(pf4Points2.USize());

		TBuffer<float> pfDists;
		pfDists.alloc(pf4Points1.USize());

		float fDist1To2;

		//////////////////////////// test GPU //////////////////////////////
		_GPUDistUseCpu(false);	// GPU
		for(int i = 0; i < iNrOfTrials; i++)
		{
			_GPUDistCompDistFromPointsToPoints
			(
				pf4Points1.USize(),
				&pf4Points1[0],

				pf4Points2.USize(),
				&pf4Points2[0],

				&pfDists[0]
			);

			fDist1To2 = -(float)HUGE_VAL;
			for(size_t p = 0; p < pfDists.USize(); p++)
				fDist1To2 = max(fDist1To2, pfDists[p]);
			fDist1To2 = sqrtf(fDist1To2);
		}
		LOG_VAR(fDist1To2);

		//////////////////////////// test CPU //////////////////////////////
		_GPUDistUseCpu(true);	// CPU
		for(int i = 0; i < iNrOfTrials; i++)
		{
			_GPUDistCompDistFromPointsToPoints
			(
				pf4Points1.USize(),
				&pf4Points1[0],

				pf4Points2.USize(),
				&pf4Points2[0],

				&pfDists[0]
			);

			fDist1To2 = -(float)HUGE_VAL;
			for(size_t p = 0; p < pfDists.USize(); p++)
				fDist1To2 = max(fDist1To2, pfDists[p]);
			fDist1To2 = sqrtf(fDist1To2);
		}
		LOG_VAR(fDist1To2);
	}

	return 0;
}