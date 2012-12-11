#define	_USE_MATH_DEFINES	// ADD-BY-LEETEN 04/13/2012
#include <math.h>	// ADD-BY-LEETEN 04/07/2012
#include <vector_functions.h>

#include "libclock.h"	// ADD-BY-LEETEN 04/07/2012
#include "libopt.h"	// ADD-BY-LEETEN 04/14/2012
#include "liblog.h"
#include "libbuf.h"
#include "libbuf3d.h"

#include "GPUDistLib.h"

#if	0	// MOD-BY-LEETEN 04/17/2012-FROM:
	#define TEST_MESH_TETRAHEDRON	1
	#define TEST_MESH_TUBE		2
	#define TEST_MESH_SPHERE	3	
	#define TEST_MESH		TEST_MESH_SPHERE
#else		// MOD-BY-LEETEN 04/17/2012-TO:
enum
{
	TEST_MESH_TETRAHEDRON,
	TEST_MESH_TUBE,
	TEST_MESH_SPHERE,
	TEST_MESH_OFF,	// ADD-BY-LEETEN 12/10/2012
	// ADD-BY-LEETEN 04/24/2012-BEGIN
	TEST_MESH_ISO_VERTICES,	
	TEST_MESH_ISO_TRIANGLES,
	// ADD-BY-LEETEN 04/24/2012-END

	NR_OF_TEST_MESHES
};
#endif		// MOD-BY-LEETEN 04/17/2012-END

int 
main(int argn, char* argv[])
{
	// ADD-BY-LEETEN 04/14/2012-BEGIN
	_OPTInit();

	int iNrOfSlices;
	_OPTAddIntegerVector("--nr-of-slices", 1, 
		&iNrOfSlices, 16);

	int piGridSize[3];
	_OPTAddIntegerVector("--grid-size", 3, 
		&piGridSize[0], 16,
		&piGridSize[1], 16,
		&piGridSize[2], 16);
	char* szDistanceFieldFilenamePrefix;
	_OPTAddStringVector("--distance-field-filename-prefix", 1, 
		&szDistanceFieldFilenamePrefix, "distance_field");

	// ADD-BY-LEETEN 04/14/2012-BEGIN
	int iIsUsingCpu;
	_OPTAddBoolean("--is-using-cpu", &iIsUsingCpu, OPT_FALSE);

	int iTestMesh;
	// MOD-BY-LEETEN 04/17/2012-FROM:	_OPTAddEnum("--test-mesh", &iTestMesh, 3, TEST_MESH_SPHERE,
	_OPTAddEnum("--test-mesh", &iTestMesh, TEST_MESH_SPHERE, NR_OF_TEST_MESHES, 
	// MOD-BY-LEETEN 04/17/2012-END
		"tetrahedron",	TEST_MESH_TETRAHEDRON,
		"tube",		TEST_MESH_TUBE,
		// MOD-BY-LEETEN 04/24/2012-FROM:		"sphere",	TEST_MESH_SPHERE);
		"sphere",	TEST_MESH_SPHERE,
		"off",		TEST_MESH_OFF,		// ADD-BY-LEETEN 12/10/2012
		"iso-vertices",	TEST_MESH_ISO_VERTICES,
		"iso-triangles",TEST_MESH_ISO_TRIANGLES
		);
		// MOD-BY-LEETEN 04/24/2012-END

	// ADD-BY-LEETEN 12/10/2012-BEGIN
	char *szOffFilePath;
	_OPTAddStringVector("--off-filepath", 1, &szOffFilePath);

	int iIsUsingAbsCoords;
	_OPTAddBoolean("--is-using-abs-coords", &iIsUsingAbsCoords, OPT_FALSE);
	// ADD-BY-LEETEN 12/10/2012-END

	// ADD-BY-LEETEN 04/24/2012-BEGIN
	char *szVertexHeaderFilePath;
	char *szVertexDataFilePath;
	char *szTrianglesHeaderFilePath;
	char *szTriangleDataFilePath;
	_OPTAddStringVector("--iso-mesh", 4,
		&szVertexHeaderFilePath, NULL,
		&szVertexDataFilePath, NULL,
		&szTrianglesHeaderFilePath, NULL,
		&szTriangleDataFilePath, NULL
		);
	// ADD-BY-LEETEN 04/24/2012-END

	// ADD-BY-LEETEN 04/14/2012-END

	// ADD-BY-LEETEN 04/15/2012-BEGIN
	int iIsPreCompTransforms;
	_OPTAddBoolean("--is-pre-comp-transforms", &iIsPreCompTransforms, OPT_FALSE);
	// ADD-BY-LEETEN 04/15/2012-END

	ASSERT_OR_LOG(
		BOPTParse(argv, argn, 1), 
		printf("Invalid Arguments."));

	LOG_VAR(szDistanceFieldFilenamePrefix);
	LOG_VAR(iNrOfSlices);
	LOG(printf("Grid Size = %d %d %d", piGridSize[0], piGridSize[1], piGridSize[2]));
	// MOD-BY-LEETEN 04/24/2012-FROM:	char *szDevice = (iIsUsingCpu)?"cpu":"gpu";
	const char *szDevice = (iIsUsingCpu)?"cpu":"gpu";
	// MOD-BY-LEETEN 04/24/2012-END
	LOG_VAR(szDevice);
	// ADD-BY-LEETEN 04/14/2012-END
	_GPUDistInit();	// ADD-BY-LEETEN 04/07/2012

	_GPUDistPrintTiming(true);

	// MOD-BY-LEETEN 04/15/2012:	bool bIsPreCompTransforms = true;	// ADD-BY-LEETEN 04/14/2012
	bool bIsPreCompTransforms = iIsPreCompTransforms;	
	// MOD-BY-LEETEN 04/15/2012-END

	// build the grid
	// DEL-BY-LEETEN 04/14/2012:	for(int testi = 0; testi < 1; testi++)
	{
		// MOD-BY-LEETEN 04/14/2012-FROM::	_GPUDistUseCpu(testi);
		_GPUDistUseCpu(iIsUsingCpu);
		// MOD-BY-LEETEN 04/14/2012-END

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
		// DEL-BY-LEETEN 04/14/2012:	size_t uLength = 32;
		#endif		// MOD-BY-LEETEN 04/13/2012-END

		// ADD-BY-LEETEN 04/14/2012-BEGIN
		TBuffer<float4> pf4Vertices;
		TBuffer<ulong4> pu4TriangleVertices;
		switch(iTestMesh)
		{
		// ADD-BY-LEETEN 12/10/2012-BEGIN
		case TEST_MESH_OFF:
			{
			FILE *fp;

			ASSERT_OR_LOG(NULL !=		szOffFilePath,		fprintf(stderr, "Missing szOffFilePath"));
			ASSERT_OR_LOG(fp = fopen(	szOffFilePath, "rt"),	perror(szOffFilePath));
			char szHeader[1024];
			fgets(szHeader, sizeof(szHeader), fp);	// skip the first line, which is the header OFF
			int iNrOfVertices;	fscanf(fp, "%d", &iNrOfVertices);
			int iNrOfTriangles;	fscanf(fp, "%d", &iNrOfTriangles);
			int iNonUsed;		fscanf(fp, "%d", &iNonUsed);

			pf4Vertices.alloc(iNrOfVertices);
			for(size_t v = 0; v < pf4Vertices.USize(); v++)
			{
				fscanf(fp, "%f %f %f", &pf4Vertices[v].x, &pf4Vertices[v].y, &pf4Vertices[v].z);
				if( !iIsUsingAbsCoords )
				{
					pf4Vertices[v].x /= (float)(piGridSize[0] - 1);
					pf4Vertices[v].y /= (float)(piGridSize[1] - 1);
					pf4Vertices[v].z /= (float)(piGridSize[2] - 1);
				}
				pf4Vertices[v].w = 1.0f;
			}

			pu4TriangleVertices.alloc(iNrOfTriangles);
			for(size_t t = 0; t < pu4TriangleVertices.USize(); t++)
			{
				int iN;
				fscanf(fp, "%d", &iN);
				for(size_t v = 0; v < (size_t)iN; v++)
				{
					int iI;
					fscanf(fp, "%d", &iI);
					switch(v){
					case 0:	pu4TriangleVertices[t].x = iI;	break;
					case 1:	pu4TriangleVertices[t].y = iI;	break;
					case 2:	pu4TriangleVertices[t].z = iI;	break;
					}
				}
			}

			fclose(fp);
			} break;
		// ADD-BY-LEETEN 12/10/2012-END

		// ADD-BY-LEETEN 04/24/2012-BEGIN
		case TEST_MESH_ISO_TRIANGLES:
			{
			FILE *fp;

			ASSERT_OR_LOG(NULL !=		szVertexHeaderFilePath,		fprintf(stderr, "Missing szVertexHeaderFilePath"));
			ASSERT_OR_LOG(fp = fopen(	szVertexHeaderFilePath, "rt"),	perror(szVertexHeaderFilePath));
			int iNrOfVertices;	fscanf(fp, "%d", &iNrOfVertices);	fclose(fp);
			pf4Vertices.alloc(iNrOfVertices);

			TBuffer<float> pfVertices;	
			pfVertices.alloc(3 * iNrOfVertices);

			ASSERT_OR_LOG(NULL !=		szVertexDataFilePath,		fprintf(stderr, "Missing szVertexDataFilePath"));
			ASSERT_OR_LOG(fp = fopen(	szVertexDataFilePath, "rb"),	perror(szVertexDataFilePath));
			fread(&pfVertices[0], sizeof(pfVertices[0]), pfVertices.USize(), fp);	fclose(fp);
			for(size_t v = 0; v < pf4Vertices.USize(); v++)
			{
				pf4Vertices[v] = make_float4(
					pfVertices[v*3+0] / (float)(piGridSize[0] - 1),
					pfVertices[v*3+1] / (float)(piGridSize[1] - 1),
					pfVertices[v*3+2] / (float)(piGridSize[2] - 1),
					1.0f);
			}

			ASSERT_OR_LOG(NULL !=		szTrianglesHeaderFilePath,	fprintf(stderr, "Missing szTrianglesHeaderFilePath"));
			ASSERT_OR_LOG(fp = fopen(	szTrianglesHeaderFilePath, "rt"), perror(szTrianglesHeaderFilePath));
			int iNrOfTriangles;	fscanf(fp, "%d", &iNrOfTriangles);	fclose(fp);
			pu4TriangleVertices.alloc(iNrOfTriangles);

			TBuffer<int> piTriangles;	
			piTriangles.alloc(3 * iNrOfTriangles);

			ASSERT_OR_LOG(NULL !=		szTriangleDataFilePath,		fprintf(stderr, "Missing szTriangleDataFilePath"));
			ASSERT_OR_LOG(fp = fopen(	szTriangleDataFilePath, "rb"),	perror(szTriangleDataFilePath));
			fread(&piTriangles[0], sizeof(piTriangles[0]), piTriangles.USize(), fp);	fclose(fp);
			for(size_t t = 0; t < pu4TriangleVertices.USize(); t++)
			{
				// LOG(printf("%d %d %d", piTriangles[t*3+0], piTriangles[t*3+1], piTriangles[t*3+2]));
				pu4TriangleVertices[t] = make_ulong4(
					piTriangles[t*3+0],
					piTriangles[t*3+1],
					piTriangles[t*3+2],
					0);
			}
			}	break;

		case TEST_MESH_ISO_VERTICES:
			{
			FILE *fp;

			ASSERT_OR_LOG(NULL !=		szVertexHeaderFilePath,		fprintf(stderr, "Missing szVertexHeaderFilePath"));
			ASSERT_OR_LOG(fp = fopen(	szVertexHeaderFilePath, "rt"),	perror(szVertexHeaderFilePath));
			int iNrOfVertices;	fscanf(fp, "%d", &iNrOfVertices);	fclose(fp);
			pf4Vertices.alloc(iNrOfVertices);

			TBuffer<float> pfVertices;	
			pfVertices.alloc(3 * iNrOfVertices);

			ASSERT_OR_LOG(NULL !=		szVertexDataFilePath,		fprintf(stderr, "Missing szVertexDataFilePath"));
			ASSERT_OR_LOG(fp = fopen(	szVertexDataFilePath, "rb"),	perror(szVertexDataFilePath));
			fread(&pfVertices[0], sizeof(pfVertices[0]), pfVertices.USize(), fp);	fclose(fp);
			for(size_t v = 0; v < pf4Vertices.USize(); v++)
			{
				pf4Vertices[v] = make_float4(
					pfVertices[v*3+0] / (float)(piGridSize[0] - 1),
					pfVertices[v*3+1] / (float)(piGridSize[1] - 1),
					pfVertices[v*3+2] / (float)(piGridSize[2] - 1),
					1.0f);
			}

			int iNrOfTriangles = pf4Vertices.USize()/3;
			pu4TriangleVertices.alloc(iNrOfTriangles);
			for(size_t t = 0; t < pu4TriangleVertices.USize(); t++)
			{
				// LOG(printf("%d %d %d", piTriangles[t*3+0], piTriangles[t*3+1], piTriangles[t*3+2]));
				pu4TriangleVertices[t] = make_ulong4(
					t*3+0,
					t*3+1,
					t*3+2,
					0);
			}
			}	break;
		// ADD-BY-LEETEN 04/24/2012-END
		case	TEST_MESH_TETRAHEDRON:
			{
		// ADD-BY-LEETEN 04/14/2012-END

			// create a small triangle meshes
			// DEL-BY-LEETEN 04/14/2012:	#if	TEST_MESH == TEST_MESH_TETRAHEDRON
			// a small tetrahedron
			// DEL-BY-LEETEN 04/14/2012:	TBuffer<float4> pf4Vertices;
			pf4Vertices.alloc(4);
			#if	0	// MOD-BY-LEETEN 04/14/2012-FROM:
				pf4Vertices[0] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
				pf4Vertices[1] = make_float4((float)uLength - 2.0f, (float)uLength - 2.0f, 2.0f, 1.0f);
				pf4Vertices[2] = make_float4((float)uLength - 2.0f, 2.0f, (float)uLength - 2.0f, 1.0f);
				pf4Vertices[3] = make_float4(2.0f, (float)uLength - 2.0f, (float)uLength - 2.0f, 1.0f);
			#else		// MOD-BY-LEETEN 04/14/2012-TO:
			pf4Vertices[0] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			pf4Vertices[1] = make_float4(1.0f, 1.0f, 0.0f, 1.0f);
			pf4Vertices[2] = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
			pf4Vertices[3] = make_float4(0.0f, 1.0f, 1.0f, 1.0f);
			#endif		// MOD-BY-LEETEN 04/14/2012-END

			// DEL-BY-LEETEN 04/14/2012:	TBuffer<ulong4> pu4TriangleVertices;
			pu4TriangleVertices.alloc(4);
			pu4TriangleVertices[0] = make_ulong4(0, 1, 2, 0);
			pu4TriangleVertices[1] = make_ulong4(0, 1, 3, 0);
			pu4TriangleVertices[2] = make_ulong4(0, 2, 3, 0);
			pu4TriangleVertices[3] = make_ulong4(1, 2, 3, 0);
		// ADD-BY-LEETEN 04/14/2012-BEGIN
			}	break;

		case	TEST_MESH_TUBE:
			{
		// ADD-BY-LEETEN 04/14/2012-END

			// DEL-BY-LEETEN 04/14/2012:	#elif	TEST_MESH == TEST_MESH_TUBE,
			// a tube
			double dRadius = 0.25;
			float4 f4Origin = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
			#if	0	// MOD-BY-LEETEN 04/14/2012-FROM:
				TBuffer<float4> pf4Vertices;
				pf4Vertices.alloc( (uLength + 1) * 2 );
				TBuffer<ulong4> pu4TriangleVertices;
				pu4TriangleVertices.alloc( uLength * 2 );
				for(size_t p = 0, l = 0; l < uLength  + 1; l++, p+=2)
				{
				double dTheta = 2.0 * M_PI * (double)l / (double)uLength;
			#else		// MOD-BY-LEETEN 04/14/2012-TO:
			size_t uNrOfSlices = iNrOfSlices;
			pf4Vertices.alloc( (uNrOfSlices + 1) * 2 );
			pu4TriangleVertices.alloc( uNrOfSlices * 2 );
			for(size_t p = 0, l = 0; l < uNrOfSlices + 1; l++, p+=2)
			{
				double dTheta = 2.0 * M_PI * (double)l / (double)iNrOfSlices;
			#endif		// MOD-BY-LEETEN 04/14/2012-END
				double dX = cos(dTheta) * dRadius;
				double dY = sin(dTheta) * dRadius;
				pf4Vertices[p]		= make_float4(f4Origin.x + (float)dX, f4Origin.y + (float)dY, 0.0f, 1.0f);
				pf4Vertices[p+1]	= make_float4(f4Origin.x + (float)dX, f4Origin.y + (float)dY, 1.0f, 1.0f);
				// MOD-BY-LEETEN 04/14/2012-FROM:				if( l < uLength )
				if( l < uNrOfSlices )
				// MOD-BY-LEETEN 04/14/2012-END
				{
					pu4TriangleVertices[p] =	make_ulong4(p, p+2, p+3, 0);
					pu4TriangleVertices[p + 1] =	make_ulong4(p, p+3, p+1, 0);
				}
			}
		// ADD-BY-LEETEN 04/14/2012-BEGIN
			}	break;

		case	TEST_MESH_SPHERE:
			{
		// ADD-BY-LEETEN 04/14/2012-END
			// DEL-BY-LEETEN 04/14/2012:	#elif	TEST_MESH == TEST_MESH_SPHERE,
			// a sphere
			float4 f4Origin = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
			float fRadius = 0.25f;
			#if	0	// MOD-BY-LEETEN 04/14/2012-FROM:
				size_t uNrOfSlices = 2 * uLength;
				size_t uNrOfStacks = 2 * uLength;
			#else		// MOD-BY-LEETEN 04/14/2012-TO:
			size_t uNrOfSlices = (size_t)iNrOfSlices;
			size_t uNrOfStacks = uNrOfSlices;
			#endif		// MOD-BY-LEETEN 04/14/2012-END	
			// DEL-BY-LEETEN 04/14/2012:	TBuffer<float4> pf4Vertices;
			pf4Vertices.alloc( (uNrOfSlices + 1) * (uNrOfStacks + 1) );
			// DEL-BY-LEETEN 04/14/2012:	TBuffer<ulong4> pu4TriangleVertices;
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
			// DEL-BY-LEETEN 04/14/2012:	#endif	// #if TEST_MESH
		// ADD-BY-LEETEN 04/14/2012-BEGIN
			}	break;
		}
		// ADD-BY-LEETEN 04/14/2012-END

		// ADD-BY-LEETEN 04/14/2012-BEGIN
		LOG_VAR(pf4Vertices.USize());
		LOG_VAR(pu4TriangleVertices.USize());
		// ADD-BY-LEETEN 04/14/2012-END

		// ADD-BY-LEETEN 04/13/2012-BEGIN
		// DEL-BY-LEETEN 04/14/2012:	for(size_t le = 5; le < 6; le++)
		{
			TBuffer3D<float> p3DfDist;
			#if	0	// MOD-BY-LEETEN 04/14/2012-FROM:
				size_t uLength = 1 << le;
				LOG_VAR(uLength);
				p3DfDist.alloc(uLength, uLength, uLength);
			#else		// MOD-BY-LEETEN 04/14/2012-TO:
			p3DfDist.alloc(piGridSize[0], piGridSize[1], piGridSize[2]);
			#endif		// MOD-BY-LEETEN 04/14/2012-END

			TBuffer<float4> pf4Coords;
			pf4Coords.alloc(p3DfDist.USize());
			for(size_t	v = 0,		d = 0; d < p3DfDist.iDepth; d++)
				for(size_t		h = 0; h < p3DfDist.iHeight; h++)
					for(size_t	w = 0; w < p3DfDist.iWidth; w++, v++)
						// ADD-BY-LEETEN 12/10/2012-BEGIN
						if(iIsUsingAbsCoords)
							pf4Coords[v] = make_float4((float)w, (float)h, (float)d, 1.0f);
						else
						// ADD-BY-LEETEN 12/10/2012-END
						pf4Coords[v] = make_float4(
							(float)w / (float)(p3DfDist.iWidth - 1), (float)h / (float)(p3DfDist.iHeight - 1), (float)d / (float)(p3DfDist.iDepth - 1), 1.0f);

			LOG_VAR(pf4Coords.USize());	// ADD-BY-LEETEN 04/14/2012

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
			// MOD-BY-LEETEN 04/14/2012-FROM:			sprintf(szFileName, "%s.point.dist", (0 == testi)?"gpu":"cpu");
			sprintf(szFileName, "%s.%s.point.dist", szDistanceFieldFilenamePrefix, szDevice);
			// MOD-BY-LEETEN 04/14/2012-END

			p3DfDist._Save(szFileName);

			#if	0	// DEL-BY-LEETEN 04/15/2012-BEGIN
				// ADD-BY-LEETEN 04/14/2012-BEGIN
				// compute the distance from the points to the triangle meshes
				_GPUDistCompDistFromPointsToTriangles
				(
					pf4Coords.USize(),
					&pf4Coords[0],

					pf4Vertices.USize(),
					&pf4Vertices[0],

					pu4TriangleVertices.USize(),
					&pu4TriangleVertices[0],

					!bIsPreCompTransforms,

					&p3DfDist[0]
				);
				// ADD-BY-LEETEN 04/14/2012-END
			#endif		// DEL-BY-LEETEN 04/15/2012-END

			// compute the distance from the points to the triangle meshes
			_GPUDistCompDistFromPointsToTriangles
			(
				pf4Coords.USize(),
				&pf4Coords[0],

				pf4Vertices.USize(),
				&pf4Vertices[0],

				pu4TriangleVertices.USize(),
				&pu4TriangleVertices[0],

				// MOD-BY-LEETEN 04/14/2012-FROM:				false,
				bIsPreCompTransforms,
				// MOD-BY-LEETEN 04/14/2012-END

				&p3DfDist[0]
			);

			// ADD-BY-LEETEN 04/13/2012-BEGIN
			if( !BGPUDistIsDistSquaredRoot() )
				for(unsigned int i = 0; i < p3DfDist.USize(); i++)
				// ADD-BY-LEETEN 04/17/2012-BEGIN
				{
				// ADD-BY-LEETEN 04/17/2012-END
					p3DfDist[i] = sqrtf(p3DfDist[i]);
				}	// ADD-BY-LEETEN 04/17/2012
			// ADD-BY-LEETEN 04/13/2012-END
			// MOD-BY-LEETEN 04/14/2012-FROM:		sprintf(szFileName, "%s.triangle.dist", (0 == testi)?"gpu":"cpu");
			sprintf(szFileName, "%s.%s.triangle.dist", szDistanceFieldFilenamePrefix, szDevice);
			// MOD-BY-LEETEN 04/14/2012-END
			p3DfDist._Save(szFileName);
		}
	}
	return 0;
}