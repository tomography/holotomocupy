#include "ptychofft.cuh"
#include "kernels_ptycho.cu"

ptychofft::ptychofft(size_t ptheta, size_t nz, size_t n,
	size_t nscan, size_t ndet, size_t nprb, size_t ngpus)
	: ptheta(ptheta), nz(nz), n(n), nscan(nscan), ndet(ndet), nprb(nprb), ngpus(ngpus) {	
	
	int ffts[2];
	int idist;int odist;
	int inembed[2];int onembed[2];
	ffts[0] = ndet; ffts[1] = ndet;
	idist = ndet*ndet; odist = ndet*ndet;
	inembed[0] = ndet; inembed[1] = ndet;
	onembed[0] = ndet; onembed[1] = ndet;
	plan2d = new cufftHandle[ngpus];
	shiftx = new float2*[ngpus];
	shifty = new float2*[ngpus];
	for (int igpu=0;igpu<ngpus;igpu++)
  	{
    	cudaSetDevice(igpu);
		cudaMalloc((void**)&shiftx[igpu],ptheta*nscan*sizeof(float2));
		cudaMalloc((void**)&shifty[igpu],ptheta*nscan*sizeof(float2));
		cufftPlanMany(&plan2d[igpu], 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_C2C, ptheta*nscan); 
	}
	cudaSetDevice(0);
}

ptychofft::~ptychofft(){free();}

void ptychofft::free()
{	
	if (!is_free) 
	{
		for (int igpu=0;igpu<ngpus;igpu++)
  		{
			cudaSetDevice(igpu);
			cudaFree(shiftx[igpu]);
			cudaFree(shifty[igpu]);
			cufftDestroy(plan2d[igpu]);
		}
		is_free = true;   
		cudaSetDevice(0);
	}	
}


void ptychofft::fwd(size_t g_, size_t f_, size_t prb_, size_t scan_, size_t igpu)
{
	cudaSetDevice(igpu);
	
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(nprb*nprb/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ptheta/(float)BS3d.z));
	dim3 GS3d1(ceil(ndet*ndet/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ptheta/(float)BS3d.z));
	dim3 GS2d0(ceil(nscan/(float)BS3d.x),ceil(ptheta/(float)BS3d.y));
	
	float2* f = (float2*)f_;
	float2* g = (float2*)g_;
	float2* prb = (float2*)prb_;
	float* scany = (float*)&((float*)scan_)[0];
	float* scanx = (float*)&((float*)scan_)[ptheta*nscan];
	mul<<<GS3d0,BS3d>>>(g,f,prb,scanx,scany,ptheta,nz,n,nscan,nprb,ndet,ndet);	
	cufftExecC2C(plan2d[igpu], (cufftComplex*)g,(cufftComplex*)g,CUFFT_FORWARD);
	takeshifts<<<GS2d0,BS3d>>>(shiftx[igpu],shifty[igpu],scanx,scany,ptheta,nscan);		
	shifts<<<GS3d1,BS3d>>>(g, shiftx[igpu], shifty[igpu], ptheta, nscan, ndet*ndet);
}

void ptychofft::adj(size_t f_, size_t g_, size_t prb_, size_t scan_, size_t igpu)
{
	cudaSetDevice(igpu);
	
	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(nprb*nprb/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ptheta/(float)BS3d.z));
	dim3 GS3d1(ceil(ndet*ndet/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ptheta/(float)BS3d.z));
	dim3 GS2d0(ceil(nscan/(float)BS3d.x),ceil(ptheta/(float)BS3d.y));
	
	float2* f = (float2*)f_;
	float2* g = (float2*)g_;
	float2* prb = (float2*)prb_;
	float* scany = (float*)&((float*)scan_)[0];
	float* scanx = (float*)&((float*)scan_)[ptheta*nscan];
	
	takeshifts<<<GS2d0,BS3d>>>(shiftx[igpu],shifty[igpu],scanx,scany,ptheta,nscan);		
	shiftsa<<<GS3d1,BS3d>>>(g, shiftx[igpu], shifty[igpu], ptheta, nscan, ndet*ndet);
	cufftExecC2C(plan2d[igpu], (cufftComplex*)g,(cufftComplex*)g,CUFFT_INVERSE);
	mula<<<GS3d0,BS3d>>>(f,g,prb,scanx,scany,ptheta,nz,n,nscan,nprb,ndet,ndet);	
}

void ptychofft::adjprb(size_t prb_, size_t g_, size_t f_, size_t scan_, size_t igpu)
{
	cudaSetDevice(igpu);

	dim3 BS3d(32,32,1);
	dim3 GS3d0(ceil(nprb*nprb/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ptheta/(float)BS3d.z));
	dim3 GS3d1(ceil(ndet*ndet/(float)BS3d.x),ceil(nscan/(float)BS3d.y),ceil(ptheta/(float)BS3d.z));
	dim3 GS2d0(ceil(nscan/(float)BS3d.x),ceil(ptheta/(float)BS3d.y));
	
	float2* f = (float2*)f_;
	float2* g = (float2*)g_;
	float2* prb = (float2*)prb_;
	float* scany = (float*)&((float*)scan_)[0];
	float* scanx = (float*)&((float*)scan_)[ptheta*nscan];
	
	takeshifts<<<GS2d0,BS3d>>>(shiftx[igpu],shifty[igpu],scanx,scany,ptheta,nscan);		
	shiftsa<<<GS3d1,BS3d>>>(g, shiftx[igpu], shifty[igpu], ptheta, nscan, ndet*ndet);
	cufftExecC2C(plan2d[igpu], (cufftComplex*)g,(cufftComplex*)g,CUFFT_INVERSE);
	mulaprb<<<GS3d0,BS3d>>>(f,g,prb,scanx,scany,ptheta,nz,n,nscan,nprb,ndet,ndet);
}







