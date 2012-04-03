#include <vector_functions.h>

#include "liblog.h"
#include "libbuf.h"
#include "libbuf3d.h"

#include "GPUDistLib.h"

int 
main(int argn, char* argv[])
{
	TBuffer3D<float> p3DfDist;
	for(int testi = 0, le = 4; le < 7; le++)
	{
		size_t uLength = 1<<le;
		p3DfDist.alloc(uLength, uLength, uLength);

		for(int pe = 0; pe < 4; pe++, testi++)
		{
			TBuffer<float4> pf4Points;
			pf4Points.alloc( (int)((float)(uLength * uLength) * (0.5 + pe) / 4.0f) );
			for(size_t p = 0; p < pf4Points.USize(); p++)
			{
				pf4Points[p] = make_float4(
					(float)(rand() % p3DfDist.iWidth),
					(float)(rand() % p3DfDist.iHeight),
					(float)(rand() % p3DfDist.iDepth),
					1.0f);
				// 		LOG(printf("%.2f %.2f %.2f", pf4Points[p].x, pf4Points[p].y, pf4Points[p].z));
			}
			LOG(printf("#Voxels = %d^3, #Points = %d", uLength, pf4Points.USize()));

			_GPUDistPrintTiming(true);

			// GPU
			_GPUDistUseCpu(false);
			if( !testi )
				_GPUDistComputeDistanceFieldFromPoints
				(
					pf4Points.USize(),
					&pf4Points[0],
					p3DfDist.iWidth,
					p3DfDist.iHeight,
					p3DfDist.iDepth,
					&p3DfDist[0]
				);

			_GPUDistComputeDistanceFieldFromPoints
			(
				pf4Points.USize(),
				&pf4Points[0],
				p3DfDist.iWidth,
				p3DfDist.iHeight,
				p3DfDist.iDepth,
				&p3DfDist[0]
			);
		}
	}
	// p3DfDist._Save("dist");
	return 0;
}