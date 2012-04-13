#pragma once

__device__
float4
F4Cross_device(float4 f4A, float4 f4B)
{
	return make_float4(
		f4A.y * f4B.z - f4A.z * f4B.y, // c[0]=a[1]*b[2] - a[2]*b[1];
		f4A.z * f4B.x - f4A.x * f4B.z, // c[1]=a[2]*b[0] - a[0]*b[2],
		f4A.x * f4B.y - f4A.y * f4B.x, // c[2]=a[0]*b[1] - a[1]*b[0],
		f4A.w);
}
__device__
float4
F4Normalize_device(float4 f4V)
{
	float4 f4V2 = f4V;
	float fL = sqrtf(f4V.x * f4V.x + f4V.y * f4V.y + f4V.z * f4V.z);
	if( fL > 0.0f )
		f4V2 = make_float4(f4V2.x / fL, f4V2.y / fL, f4V2.z / fL, f4V2.w);
	return f4V2;
}

__device__
float
FDot_device(float4 f4V1, float4 f4V2)
{
	return f4V1.x * f4V2.x + f4V1.y * f4V2.y + f4V1.z * f4V2.z;
}

/*

$Log$

*/
