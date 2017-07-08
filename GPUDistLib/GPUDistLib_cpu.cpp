#include "GPUDistLib_internal.h"

#include <vector_functions.h>
#include <lib3ds/vector.h>

void
_CompDistToEdge2D
(
	float2 f2P,
	float2 f2A,
	float2 f2B,

	float *pfDist
)
{
	/*
	Lib3dsVector v3AP;
	ib3ds_vector_sub(v3AP, v3A, v3P);
	// float fAP = lib3ds_vector_length( v3AP );
	*/
	float2 f2AP;
	f2AP.x = f2P.x - f2A.x;
	f2AP.y = f2P.y - f2A.y;
	float fAP = f2AP.x * f2AP.x + f2AP.y * f2AP.y;
	#if	IS_SQRT
	fAP = sqrtf(fAP);
	#endif	// #if	IS_SQRT

	/*
	Lib3dsVector v3AB;
	ib3ds_vector_sub(v3AB, v3A, v3B);
	*/
	float2 f2AB;
	f2AB.x = f2B.x - f2A.x;
	f2AB.y = f2B.y - f2A.y;
	float fAB = sqrtf(f2AB.x * f2AB.x + f2AB.y * f2AB.y);

	float fD = fAP;
	if( 0.0f < fAB )
	{
		// float fT = lib3ds_vector_dot(v3AB, v3AP) / fAB;	// fT: |AP| cos(theta)
		float fT = (f2AB.x * f2AP.x + f2AB.y * f2AP.y) / fAB;
		if( 0.0f <= fT && fT <= fAB )
		{
			#if	IS_SQRT
			fD = sqrtf(fAP * fAP - fT * fT);
			#else	// #if	IS_SQRT		
			fD = fAP - fT * fT;
			#endif	// #if	IS_SQRT
		}
		else
		{
			/*
			ib3ds_vector_sub(v3BP, v3B, v3P);
			float fBP = lib3ds_vector_length( v3BP );
			*/
			float2 f2BP;
			f2BP.x = f2P.x - f2B.x;
			f2BP.y = f2P.y - f2B.y;
			#if	IS_SQRT
			float fBP = sqrtf(f2BP.x * f2BP.x + f2BP.y * f2BP.y);
			#else	// #if	IS_SQRT
			float fBP = f2BP.x * f2BP.x + f2BP.y * f2BP.y;
			#endif	// #if	IS_SQRT

			fD = min(fAP, fBP);
		}
	}
	*pfDist = fD;
}

void
_CompDistToEdge
(
	Lib3dsVector v3P,
	Lib3dsVector v3A,
	Lib3dsVector v3B,

	float *pfDist
)
{
	Lib3dsVector v3AP;
	lib3ds_vector_sub(v3AP, v3P, v3A);
	float fAP = v3AP[0] * v3AP[0] + v3AP[1] * v3AP[1] + v3AP[2] * v3AP[2];
	#if		IS_SQRT
	fAP = sqrtf(fAP);
	#endif	// #if	IS_SQRT

	Lib3dsVector v3AB;
	lib3ds_vector_sub(v3AB, v3B, v3A);
	float fAB = lib3ds_vector_length( v3AB );

	float fD = 0.0f;
	if( 0.0f == fAB )
	{
		fD = fAP;
	}
	else
	{
		float fT = lib3ds_vector_dot(v3AB, v3AP) / fAB;	// fT: |AP| cos(theta)
		if( 0.0f <= fT && fT <= fAB )
		{
			#if	IS_SQRT
			fD = sqrtf(fAP * fAP - fT * fT);
			#else	// #if	IS_SQRT		
			fD = fAP - fT * fT;
			#endif	// #if	IS_SQRT
		}
		else
		{
			Lib3dsVector v3BP;
			lib3ds_vector_sub(v3BP, v3P, v3B);
			float fBP = v3BP[0] * v3BP[0] + v3BP[1] * v3BP[1] + v3BP[2] * v3BP[2];
			#if	IS_SQRT
			fBP = sqrtf(fBP * fBP - fT * fT);
			#else	// #if	IS_SQRT		
			fD = fBP - fT * fT;
			#endif	// #if	IS_SQRT

			fD = min(fAP, fBP);
		}
	}
	*pfDist = fD;
}

void
_CompTransform
(
	const float4& f4A,
	const float4& f4B,
	const float4& f4C,
	Lib3dsVector v3X,
	Lib3dsVector v3Y,
	Lib3dsVector v3Z,
	Lib3dsVector v3B2,
	Lib3dsVector v3C2,
	float& fDet
 )
{
	// use A as the origin
	// use BA as the Z axis
	Lib3dsVector v3BA;
	v3BA[0] = f4B.x - f4A.x;
	v3BA[1] = f4B.y - f4A.y;
	v3BA[2] = f4B.z - f4A.z;
	lib3ds_vector_copy(v3Z, v3BA);
	lib3ds_vector_normalize(v3Z);

	// use the normal N of ABC (ie. the cross product of CA and BA) as X axis
	Lib3dsVector v3CA;
	v3CA[0] = f4C.x - f4A.x;
	v3CA[1] = f4C.y - f4A.y;
	v3CA[2] = f4C.z - f4A.z;
	lib3ds_vector_cross(v3X, v3CA, v3BA);
	lib3ds_vector_normalize(v3X);

	// use the cross product of BA and N as the y axis
	lib3ds_vector_cross(v3Y, v3BA, v3X);
	lib3ds_vector_normalize(v3Y);
	
	// transform B and C to the new coordinate
	v3B2[0] = lib3ds_vector_dot(v3BA, v3X);
	v3B2[1] = lib3ds_vector_dot(v3BA, v3Y);
	v3B2[2] = lib3ds_vector_dot(v3BA, v3Z);

	v3C2[0] = lib3ds_vector_dot(v3CA, v3X);
	v3C2[1] = lib3ds_vector_dot(v3CA, v3Y);
	v3C2[2] = lib3ds_vector_dot(v3CA, v3Z);

	fDet = v3B2[1] * v3C2[2] - v3B2[2] * v3C2[1];
}

void
_CompDistToTriangle
(
	const float4& f4P,
	const float4& f4A,
	const float4& f4B,
	const float4& f4C,
	float *pfDist,
	bool bIsPrecomputingTrasforms,
	Lib3dsVector v3PreCompX, 
	Lib3dsVector v3PreCompY, 
	Lib3dsVector v3PreCompZ,
	Lib3dsVector v3PreCompB2,
	Lib3dsVector v3PreCompC2,
	float fPreCompDet
)
{
	float fDet;
	Lib3dsVector v3X, v3Y, v3Z;
	Lib3dsVector v3B2, v3C2;
	if( !bIsPrecomputingTrasforms )
		_CompTransform
		(
			f4A, 		f4B,		f4C,
			v3X,		v3Y,		v3Z,
			v3B2,		v3C2,		fDet
		);
	else
	{
		lib3ds_vector_copy(v3X, v3PreCompX);
		lib3ds_vector_copy(v3Y, v3PreCompY);
		lib3ds_vector_copy(v3Z, v3PreCompZ);
		lib3ds_vector_copy(v3B2, v3PreCompB2);
		lib3ds_vector_copy(v3C2, v3PreCompC2);
		fDet = fPreCompDet;
	}

	float fDist = (float)HUGE_VAL;
	if( 0.0f == fDet )
	{
		Lib3dsVector v3P, v3A, v3B, v3C;
		v3P[0] = f4P.x;	v3P[1] = f4P.y;	v3P[2] = f4P.z;
		v3A[0] = f4A.x;	v3A[1] = f4A.y;	v3A[2] = f4A.z;
		v3B[0] = f4B.x;	v3B[1] = f4B.y;	v3B[2] = f4B.z;
		v3C[0] = f4C.x;	v3C[1] = f4C.y;	v3C[2] = f4C.z;

		float fD1, fD2, fD3;
		_CompDistToEdge(v3P, v3A, v3B, &fD1);
		_CompDistToEdge(v3P, v3A, v3C, &fD2);
		_CompDistToEdge(v3P, v3B, v3C, &fD3);
		fDist = min(min(fD1, fD2), fD3);
	}
	else
	{

	// and transform P to the new coordinate 
	Lib3dsVector v3PA;
	v3PA[0] = f4P.x - f4A.x;
	v3PA[1] = f4P.y - f4A.y;
	v3PA[2] = f4P.z - f4A.z;

	Lib3dsVector v3P2;
	v3P2[0] = lib3ds_vector_dot(v3PA, v3X);
	v3P2[1] = lib3ds_vector_dot(v3PA, v3Y);
	v3P2[2] = lib3ds_vector_dot(v3PA, v3Z);

	// solve (fS, fT) s.t. v3P2 = fS * v3C2 + fT * v3BA
	// v3P2[1]		v3B2[1]	v3C2[1]		fS
	//		= [			] [		]
	// v3P2[2]		v3B2[2]	v3C2[2]		fT
	// ->
	//		+v3C2[2]	-v3C2[1]	v3P2[1]		fS
	// ___	[				] [		] = [		]
	// det		-v3B2[2]	+v3B2[1]	v3P2[2]		fT
	float fS = (+v3C2[2] * v3P2[1] - v3C2[1] * v3P2[2]) / fDet;
	float fT = (-v3B2[2] * v3P2[1] + v3B2[1] * v3P2[2]) / fDet;

	fDist = fabsf(v3P2[0]);
	#if !IS_SQRT
	fDist *= fDist;
	#endif	// #if !IS_SQRT
	if( !(
		0.0f < fS && fS < 1.0f &&
		0.0f < fT && fT < 1.0f &&
		fS + fT < 1.0f	) )
	{
		float fD1, fD2, fD3;
		_CompDistToEdge2D(
			make_float2(v3P2[1], v3P2[2]),
			make_float2(0.0f, 0.0f),
			make_float2(v3B2[1], v3B2[2]),
			&fD1);
		_CompDistToEdge2D(
			make_float2(v3P2[1], v3P2[2]),
			make_float2(0.0f, 0.0f),
			make_float2(v3C2[1], v3C2[2]),
			&fD2);
		_CompDistToEdge2D(
			make_float2(v3P2[1], v3P2[2]),
			make_float2(v3B2[1], v3B2[2]),
			make_float2(v3C2[1], v3C2[2]),
			&fD3);
		float fD = min(min(fD1, fD2), fD3);

		#if	IS_SQRT
		fDist = sqrtf(fD * fD + fDist * fDist);
		#else	// #if	IS_SQRT
		fDist = fDist + fD;
		#endif	// #if IS_SQRT
	}

	}	
	*pfDist = fDist;
}

void
_GPUDistCompDistFromPointsToTrianglesByCpu
(
	size_t uNrOfPoints1,
	const float4 pf4Points1[],

	size_t uNrOfPoints2,
	const float4 pf4Points2[],

	size_t uNrOfTriangles,
	const ulong4 pu4TriangleVertices[],

	bool bIsPrecomputingTrasforms,

	float pfDists[]
)
{
	LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);

	TBuffer<Lib3dsVector> pv3Xs;
	TBuffer<Lib3dsVector> pv3Ys;
	TBuffer<Lib3dsVector> pv3Zs;
	TBuffer<Lib3dsVector> pv3B2s;
	TBuffer<Lib3dsVector> pv3C2s;
	TBuffer<float>	pfDets;
	if( bIsPrecomputingTrasforms )
	{
		pv3Xs.alloc(uNrOfTriangles);
		pv3Ys.alloc(uNrOfTriangles);
		pv3Zs.alloc(uNrOfTriangles);
		pv3B2s.alloc(uNrOfTriangles);
		pv3C2s.alloc(uNrOfTriangles);
		pfDets.alloc(uNrOfTriangles);
		for(int t = 0; t < uNrOfTriangles; t++)
		{
			_CompTransform
			(
				pf4Points2[pu4TriangleVertices[t].x],
				pf4Points2[pu4TriangleVertices[t].y],
				pf4Points2[pu4TriangleVertices[t].z],
				pv3Xs[t],
				pv3Ys[t],
				pv3Zs[t],
				pv3B2s[t],
				pv3C2s[t],
				pfDets[t]
			);
		}
	}
	LIBCLOCK_BEGIN(bIsPrintingTiming);
	for(size_t p1 = 0; p1 < uNrOfPoints1; p1++)
	{
		float fMinDist = (float)HUGE_VAL;
		for(size_t t = 0; t < uNrOfTriangles; t++)
		{
			float fDist;
			if( bIsPrecomputingTrasforms )
				_CompDistToTriangle
				(
					pf4Points1[p1],
					pf4Points2[pu4TriangleVertices[t].x],
					pf4Points2[pu4TriangleVertices[t].y],
					pf4Points2[pu4TriangleVertices[t].z],

					&fDist,

					bIsPrecomputingTrasforms,
					pv3Xs[t], pv3Ys[t], pv3Zs[t],
					pv3B2s[t], pv3C2s[t], pfDets[t]
				);
			else
				_CompDistToTriangle
				(
					pf4Points1[p1],
					pf4Points2[pu4TriangleVertices[t].x],
					pf4Points2[pu4TriangleVertices[t].y],
					pf4Points2[pu4TriangleVertices[t].z],

					&fDist,

					bIsPrecomputingTrasforms,
					NULL, NULL, NULL, 
					NULL, NULL, 0.0f
				);
			fMinDist = min(fMinDist, fDist);
		}
		pfDists[p1] = fMinDist;
	}
	LIBCLOCK_END(bIsPrintingTiming);
	LIBCLOCK_PRINT(bIsPrintingTiming);
}

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
	const float4 pf4Points1[],

	size_t uNrOfPoints2,
	const float4 pf4Points2[],

	float pfDists[],
	unsigned int *puNearestPoint2,
	void *pReserved
)
{
	LIBCLOCK_INIT(bIsPrintingTiming, __FUNCTION__);
	LIBCLOCK_BEGIN(bIsPrintingTiming);
	for(size_t p1 = 0; p1 < uNrOfPoints1; p1++)
	{
		float fMinDist = (float)HUGE_VAL;
		unsigned int uNearestPoint2 = 0;
		for(size_t p2 = 0; p2 < uNrOfPoints2; p2++)
		{
			float fDx = pf4Points2[p2].x - pf4Points1[p1].x;
			float fDy = pf4Points2[p2].y - pf4Points1[p1].y;
			float fDz = pf4Points2[p2].z - pf4Points1[p1].z;
			float fDist = fDx * fDx + fDy * fDy + fDz * fDz;
			if( fDist < fMinDist )
			{
				fMinDist = fDist;
				uNearestPoint2 = p2;
			}

		}
#if	IS_SQRT
		fMinDist = sqrtf(fMinDist);
#endif
		pfDists[p1] = fMinDist;
		if( puNearestPoint2 )
			puNearestPoint2[p1] = uNearestPoint2;
	}
	LIBCLOCK_END(bIsPrintingTiming);
	LIBCLOCK_PRINT(bIsPrintingTiming);
}


/*

$Log$

*/
