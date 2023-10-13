#include "deform.cuh"
#include <npp.h>

deform::deform(size_t ntheta, size_t nz, size_t n, size_t ptheta, size_t ngpus) :
 ntheta(ntheta), nz(nz), n(n), ptheta(ptheta), ngpus(ngpus)
{
	cstreams = new cudaStream_t[ptheta*ngpus];
	nstreams = new NppStreamContext[ptheta*ngpus];
	for (int igpu=0;igpu<ngpus;igpu++)
	{
		cudaSetDevice(igpu);
		for (int i=0;i<ptheta; i++) 
		{
			cudaStreamCreate(&cstreams[igpu*ptheta+i]);
			nstreams[igpu*ptheta+i].hStream=cstreams[igpu*ptheta+i];
		}
	}
	cudaSetDevice(0);
}

// destructor, memory deallocation
deform::~deform()
{
	free();
}

void deform::free()
{
	if (!is_free)
	{
		for (int igpu=0;igpu<ngpus;igpu++)
		{
			cudaSetDevice(igpu);
			for (int i=0;i<ptheta;i++)
			{
				cudaStreamDestroy(cstreams[igpu*ptheta+i]);
			}
		}
		delete[] cstreams;		
		delete[] nstreams;		
		
		is_free = true;
		cudaSetDevice(0);
	}
}

void deform::remap(size_t g, size_t f, size_t flowx, size_t flowy, size_t gpu)
{
	Npp32f *pSrc = (Npp32f *)f;
	NppiSize oSize = {(int)n,(int)nz};
	Npp32f *pDst = (Npp32f *)g;
	NppiRect oROI = {0,0,(int)n,(int)nz};
	int nStep = 4*n;	
	Npp32f *pXMap = (Npp32f *)flowx;
	Npp32f *pYMap = (Npp32f *)flowy;
	int nXMapStep = 4*n;
	int nYMapStep = 4*n;
	cudaSetDevice(gpu);
	for (int i=0;i<ptheta;i++)
	{
		nppiRemap_32f_C1R_Ctx(&pSrc[i*n*nz],oSize,nStep, oROI, &pXMap[i*n*nz], nXMapStep,
			 &pYMap[i*n*nz], nYMapStep, &pDst[i*n*nz], nStep, oSize, NPPI_INTER_LANCZOS,//NPPI_INTER_LANCZOS,//NPPI_INTER_LANCZOS3_ADVANCED,NPPI_INTER_CUBIC
			 nstreams[gpu*ptheta+i]);
		//nppiRemap_32f_C1R (const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, const Npp32f *pXMap, int nXMapStep, const Npp32f *pYMap, int nYMapStep, Npp32f *pDst, int nDstStep, NppiSize oDstSizeROI, int eInterpolation)
	}
	cudaDeviceSynchronize();
}