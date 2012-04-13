#define	_USE_MATH_DEFINES	// ADD-BY-LEETEN 04/13/2012
#include <math.h>	// ADD-BY-LEETEN 04/07/2012
#include <vector_functions.h>

#include "libclock.h"	// ADD-BY-LEETEN 04/07/2012
#include "liblog.h"
#include "libbuf.h"
#include "libbuf3d.h"

#include "GPUDistLib.h"

#define TEST_MESH_TETRAHEDRON	1
#define TEST_MESH_TUBE		2
#define TEST_MESH_SPHERE	3	
#define TEST_MESH		TEST_MESH_SPHERE

int 
main(int argn, char* argv[])
{
	_GPUDistInit();	// ADD-BY-LEETEN 04/07/2012

	_GPUDistPrintTiming(true);

	// build the grid
	for(int testi = 0; testi < 1; testi++)
	{
		_GPUDistUseCpu(testi);

		#if	0	// MOD-BY-LEETEN 04/13/2012-FROM:
			for(size_t le = 5; le < 6; le++)
			{
				TBuffer3D<float> p3DfDist;
				size_t uLength = 1 << le;
				LOG_VAR(uLength);
				p3DfDist.alloc(uLength, uLength, uLength);

				TBuffer<float4> pf4Coords;
				pf4Coords.alloc(p3DfDist.USize());
				for(size_t	v = 0,		d = 0; d < p3DfDist.iDepth; d++)
					for(size_t		h = 0; h < p3DfDist.iHeight; h++)
						for(size_t	w = 0; w < p3DfDist.iWidth; w++, v++)
							pf4Coords[v] = make_float4(
								(float)w / (float)(p3DfDist.iWidth - 1), (float)h / (float)(p3DfDist.iHeight - 1), (float)d / (float)(p3DfDist.iDepth - 1), 1.0f);
		#else		// MOD-BY-LEETEN 04/13/2012-TO:
		size_t uLength = 32;
		#endif		// MOD-BY-LEETEN 04/13/2012-END

			// create a small triangle meshes
			#if	TEST_MESH == TEST_MESH_TETRAHEDRON
			// a small tetrahedron
			TBuffer<float4> pf4Vertices;
			pf4Vertices.alloc(4);
			pf4Vertices[0] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
			pf4Vertices[1] = make_float4((float)uLength - 2.0f, (float)uLength - 2.0f, 2.0f, 1.0f);
			pf4Vertices[2] = make_float4((float)uLength - 2.0f, 2.0f, (float)uLength - 2.0f, 1.0f);
			pf4Vertices[3] = make_float4(2.0f, (float)uLength - 2.0f, (float)uLength - 2.0f, 1.0f);

			TBuffer<ulong4> pu4TriangleVertices;
			pu4TriangleVertices.alloc(4);
			pu4TriangleVertices[0] = make_ulong4(0, 1, 2, 0);
			pu4TriangleVertices[1] = make_ulong4(0, 1, 3, 0);
			pu4TriangleVertices[2] = make_ulong4(0, 2, 3, 0);
			pu4TriangleVertices[3] = make_ulong4(1, 2, 3, 0);
			#elif	TEST_MESH == TEST_MESH_TUBE,
			// a tube
			double dRadius = 0.25;
			float4 f4Origin = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
			TBuffer<float4> pf4Vertices;
			pf4Vertices.alloc( (uLength + 1) * 2 );
			TBuffer<ulong4> pu4TriangleVertices;
			pu4TriangleVertices.alloc( uLength * 2 );
			for(size_t p = 0, l = 0; l < uLength  + 1; l++, p+=2)
			{
				double dTheta = 2.0 * M_PI * (double)l / (double)uLength;
				double dX = cos(dTheta) * dRadius;
				double dY = sin(dTheta) * dRadius;
				pf4Vertices[p]		= make_float4(f4Origin.x + (float)dX, f4Origin.y + (float)dY, 0.0f, 1.0f);
				pf4Vertices[p+1]	= make_float4(f4Origin.x + (float)dX, f4Origin.y + (float)dY, 1.0f, 1.0f);
				if( l < uLength )
				{
					pu4TriangleVertices[p] =	make_ulong4(p, p+2, p+3, 0);
					pu4TriangleVertices[p + 1] =	make_ulong4(p, p+3, p+1, 0);
				}
			}
			#elif	TEST_MESH == TEST_MESH_SPHERE,
			// a sphere
			float4 f4Origin = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
			float fRadius = 0.25f;
			size_t uNrOfSlices = 2 * uLength;
			size_t uNrOfStacks = 2 * uLength;
			TBuffer<float4> pf4Vertices;
			pf4Vertices.alloc( (uNrOfSlices + 1) * (uNrOfStacks + 1) );
			TBuffer<ulong4> pu4TriangleVertices;
			pu4TriangleVertices.alloc( 2 * uNrOfSlices * uNrOfStacks );
			for(size_t p = 0,	stack = 0; stack < uNrOfStacks + 1; stack++)
				for(size_t	slice = 0; slice < uNrOfSlices + 1; slice++, p++)
				{
					float fTheta = 2.0f * (float)M_PI * (float)slice / (float)uNrOfSlices;
					float fPhi = (float)M_PI * (float)(stack + 1)/ (float)(uNrOfStacks + 2); // - (float)M_PI / 2.0f;
					float fZ = cosf(fPhi);
					float fX = sinf(fPhi) * cosf(fTheta);
					float fY = sinf(fPhi) * sinf(fTheta);
					pf4Vertices[p] = make_float4(
						f4Origin.x + fRadius * fX, 
						f4Origin.y + fRadius * fY, 
						f4Origin.z + fRadius * fZ, 
						1.0f);
				}

			for(size_t p = 0,	stack = 0; stack < uNrOfStacks; stack++)
				for(size_t	slice = 0; slice < uNrOfSlices; slice++, p+=2)
				{
					size_t p2 = slice + (uNrOfSlices + 1) * stack;
					pu4TriangleVertices[p] = make_ulong4(
							p2, 
							p2 + uNrOfSlices + 1, 
							p2 + 1, 
							0);
					pu4TriangleVertices[p + 1] = make_ulong4(
							p2 + 1, 
							p2 + uNrOfSlices + 1, 
							p2 + uNrOfSlices + 2, 
							0);
				}
			#endif	// #if TEST_MESH

		// ADD-BY-LEETEN 04/13/2012-BEGIN
		for(size_t le = 5; le < 6; le++)
		{
			TBuffer3D<float> p3DfDist;
			size_t uLength = 1 << le;
			LOG_VAR(uLength);
			p3DfDist.alloc(uLength, uLength, uLength);

			TBuffer<float4> pf4Coords;
			pf4Coords.alloc(p3DfDist.USize());
			for(size_t	v = 0,		d = 0; d < p3DfDist.iDepth; d++)
				for(size_t		h = 0; h < p3DfDist.iHeight; h++)
					for(size_t	w = 0; w < p3DfDist.iWidth; w++, v++)
						pf4Coords[v] = make_float4(
							(float)w / (float)(p3DfDist.iWidth - 1), (float)h / (float)(p3DfDist.iHeight - 1), (float)d / (float)(p3DfDist.iDepth - 1), 1.0f);
		// ADD-BY-LEETEN 04/13/2012-END

			////////////////////////////////////////////
			// compute the distance from the points to the mesh vertices
			_GPUDistCompDistFromPointsToPoints
			(
				pf4Coords.USize(),
				&pf4Coords[0],

				pf4Vertices.USize(),
				&pf4Vertices[0],
	 
				&p3DfDist[0]
			);
			if( !BGPUDistIsDistSquaredRoot() )	// ADD-BY-LEETEN 04/13/2012
			for(unsigned int i = 0; i < p3DfDist.USize(); i++)
				p3DfDist[i] = sqrtf(p3DfDist[i]);
			char szFileName[1024+1];
			sprintf(szFileName, "%s.point.dist", (0 == testi)?"gpu":"cpu");
			p3DfDist._Save(szFileName);

			// compute the distance from the points to the triangle meshes
			_GPUDistCompDistFromPointsToTriangles
			(
				pf4Coords.USize(),
				&pf4Coords[0],

				pf4Vertices.USize(),
				&pf4Vertices[0],

				pu4TriangleVertices.USize(),
				&pu4TriangleVertices[0],

				false,

				&p3DfDist[0]
			);
			// ADD-BY-LEETEN 04/13/2012-BEGIN
			if( !BGPUDistIsDistSquaredRoot() )
				for(unsigned int i = 0; i < p3DfDist.USize(); i++)
					p3DfDist[i] = sqrtf(p3DfDist[i]);
			// ADD-BY-LEETEN 04/13/2012-END
			sprintf(szFileName, "%s.triangle.dist", (0 == testi)?"gpu":"cpu");
			p3DfDist._Save(szFileName);
		}
	}
	return 0;
}