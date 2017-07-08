#define	_USE_MATH_DEFINES	
#include <math.h>	
#include <vector_functions.h>

#include "libclock.h"	
#include "libopt.h"	
#include "liblog.h"
#include "libbuf.h"
#include "libbuf3d.h"

#include "GPUDistLib.h"

enum
{
	TEST_MESH_TETRAHEDRON,
	TEST_MESH_TUBE,
	TEST_MESH_SPHERE,
	TEST_MESH_OFF,	
	TEST_MESH_ISO_VERTICES,	
	TEST_MESH_ISO_TRIANGLES,

	NR_OF_TEST_MESHES
};

int 
main(int argn, char* argv[])
{
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

	int iIsUsingCpu;
	_OPTAddBoolean("--is-using-cpu", &iIsUsingCpu, OPT_FALSE);

	int iTestMesh;
	_OPTAddEnum("--test-mesh", &iTestMesh, TEST_MESH_SPHERE, NR_OF_TEST_MESHES, 
		"tetrahedron",	TEST_MESH_TETRAHEDRON,
		"tube",		TEST_MESH_TUBE,
		"sphere",	TEST_MESH_SPHERE,
		"off",		TEST_MESH_OFF,		
		"iso-vertices",	TEST_MESH_ISO_VERTICES,
		"iso-triangles",TEST_MESH_ISO_TRIANGLES
		);

	char *szOffFilePath = NULL;
	_OPTAddStringVector("--off-filepath", 1, &szOffFilePath, szOffFilePath);

	int iIsUsingAbsCoords;
	_OPTAddBoolean("--is-using-abs-coords", &iIsUsingAbsCoords, OPT_FALSE);

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

	int iIsPreCompTransforms;
	_OPTAddBoolean("--is-pre-comp-transforms", &iIsPreCompTransforms, OPT_FALSE);

	ASSERT_OR_LOG(
		BOPTParse(argv, argn, 1), 
		printf("Invalid Arguments."));

	LOG_VAR(szDistanceFieldFilenamePrefix);
	LOG_VAR(iNrOfSlices);
	LOG(printf("Grid Size = %d %d %d", piGridSize[0], piGridSize[1], piGridSize[2]));
	const char *szDevice = (iIsUsingCpu)?"cpu":"gpu";
	LOG_VAR(szDevice);
	_GPUDistInit();	

	_GPUDistPrintTiming(true);

	bool bIsPreCompTransforms = iIsPreCompTransforms;	

	// build the grid
	{
		_GPUDistUseCpu(iIsUsingCpu);

		TBuffer<float4> pf4Vertices;
		TBuffer<ulong4> pu4TriangleVertices;
		switch(iTestMesh)
		{
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
		case	TEST_MESH_TETRAHEDRON:
			{

			// create a small triangle meshes
			// a small tetrahedron
			pf4Vertices.alloc(4);
			pf4Vertices[0] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			pf4Vertices[1] = make_float4(1.0f, 1.0f, 0.0f, 1.0f);
			pf4Vertices[2] = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
			pf4Vertices[3] = make_float4(0.0f, 1.0f, 1.0f, 1.0f);

			pu4TriangleVertices.alloc(4);
			pu4TriangleVertices[0] = make_ulong4(0, 1, 2, 0);
			pu4TriangleVertices[1] = make_ulong4(0, 1, 3, 0);
			pu4TriangleVertices[2] = make_ulong4(0, 2, 3, 0);
			pu4TriangleVertices[3] = make_ulong4(1, 2, 3, 0);
			}	break;

		case	TEST_MESH_TUBE:
			{
			// a tube
			double dRadius = 0.25;
			float4 f4Origin = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
			size_t uNrOfSlices = iNrOfSlices;
			pf4Vertices.alloc( (uNrOfSlices + 1) * 2 );
			pu4TriangleVertices.alloc( uNrOfSlices * 2 );
			for(size_t p = 0, l = 0; l < uNrOfSlices + 1; l++, p+=2)
			{
				double dTheta = 2.0 * M_PI * (double)l / (double)iNrOfSlices;
				double dX = cos(dTheta) * dRadius;
				double dY = sin(dTheta) * dRadius;
				pf4Vertices[p]		= make_float4(f4Origin.x + (float)dX, f4Origin.y + (float)dY, 0.0f, 1.0f);
				pf4Vertices[p+1]	= make_float4(f4Origin.x + (float)dX, f4Origin.y + (float)dY, 1.0f, 1.0f);
				if( l < uNrOfSlices )
				{
					pu4TriangleVertices[p] =	make_ulong4(p, p+2, p+3, 0);
					pu4TriangleVertices[p + 1] =	make_ulong4(p, p+3, p+1, 0);
				}
			}
			}	break;

		case	TEST_MESH_SPHERE:
			{
			// a sphere
			float4 f4Origin = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
			float fRadius = 0.25f;
			size_t uNrOfSlices = (size_t)iNrOfSlices;
			size_t uNrOfStacks = uNrOfSlices;
			pf4Vertices.alloc( (uNrOfSlices + 1) * (uNrOfStacks + 1) );
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
			}	break;
		}

		LOG_VAR(pf4Vertices.USize());
		LOG_VAR(pu4TriangleVertices.USize());

		{
			TBuffer3D<float> p3DfDist;
			p3DfDist.alloc(piGridSize[0], piGridSize[1], piGridSize[2]);

			TBuffer<float4> pf4Coords;
			pf4Coords.alloc(p3DfDist.USize());
			for(size_t	v = 0,		d = 0; d < p3DfDist.iDepth; d++)
				for(size_t		h = 0; h < p3DfDist.iHeight; h++)
					for(size_t	w = 0; w < p3DfDist.iWidth; w++, v++)
						if(iIsUsingAbsCoords)
							pf4Coords[v] = make_float4((float)w, (float)h, (float)d, 1.0f);
						else
						pf4Coords[v] = make_float4(
							(float)w / (float)(p3DfDist.iWidth - 1), (float)h / (float)(p3DfDist.iHeight - 1), (float)d / (float)(p3DfDist.iDepth - 1), 1.0f);

			LOG_VAR(pf4Coords.USize());	

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
			if( !BGPUDistIsDistSquaredRoot() )	
			for(unsigned int i = 0; i < p3DfDist.USize(); i++)
				p3DfDist[i] = sqrtf(p3DfDist[i]);
			char szFileName[1024+1];
			sprintf(szFileName, "%s.%s.point.dist", szDistanceFieldFilenamePrefix, szDevice);

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

				bIsPreCompTransforms,

				&p3DfDist[0]
			);

			if( !BGPUDistIsDistSquaredRoot() )
				for(unsigned int i = 0; i < p3DfDist.USize(); i++)
				{
					p3DfDist[i] = sqrtf(p3DfDist[i]);
				}	
			sprintf(szFileName, "%s.%s.triangle.dist", szDistanceFieldFilenamePrefix, szDevice);
			p3DfDist._Save(szFileName);
		}
	}
	return 0;
}