__global__ 
static
void 
_SetupSegments_kernel
(
	unsigned int uNrOfPoints1,
	unsigned int uNrOfPoints2,
	unsigned int puSegBegin_device[]
)
{
	unsigned int uBlock = gridDim.x * blockIdx.y + blockIdx.x;
	unsigned int uThread = uBlock * blockDim.x + threadIdx.x;

	if( uThread < uNrOfPoints1 )
		puSegBegin_device[uThread * uNrOfPoints2] = 1;
}

/*

$Log$

*/
