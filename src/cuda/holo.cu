#include "kernels_holo.cu"
#include "holo.cuh"
#include<stdio.h>
#define EPS 1e-3

holo::holo(size_t n0e_, size_t n1e_, size_t n0_, size_t n1_, size_t ntheta_) {

  n0e = n0e_; 
  n1e = n1e_;
  n0 = n0_; 
  n1 = n1_;
  ntheta = ntheta_;

  mu0 = -log(EPS) / (2 * n0e * n0e);
  mu1 = -log(EPS) / (2 * n1e * n1e);
  m0 = ceil(2 * n0e * 1 / PI * sqrt(-mu0 * log(EPS) + (mu0 * n0e) * (mu0 * n0e) / 4));
  m1 = ceil(2 * n1e * 1 / PI * sqrt(-mu1 * log(EPS) + (mu1 * n1e) * (mu1 * n1e) / 4));

  int ffts[2];
  int idist;
  int inembed[2];
  // holo 2d
  ffts[0] = 2 * n1e;
  ffts[1] = 2 * n0e;
  idist = (2 * n0e + 2 * m0) * (2 * n1e + 2 * m1);
  inembed[0] = (2 * n1e + 2 * m1);
  inembed[1] = (2 * n0e + 2 * m0);

  cudaMalloc((void **)&fdee2d, ntheta*(2 * n1e + 2 * m1) * (2 * n0e + 2 * m0) * sizeof(float2));

  cufftPlanMany(&plan2dchunk, 2, ffts, inembed, 1, idist, inembed, 1, idist, CUFFT_C2C, ntheta);

  BS2d = dim3(32, 32, 1);
  GS2d0 = dim3(ceil(n0e / (float)BS2d.x), ceil(n1e / (float)BS2d.y), ceil(ntheta / (float)BS2d.z));
  GS2d1 = dim3(ceil((2 * n0e + 2 * m0) / (float)BS2d.x), ceil((2 * n1e + 2 * m1) / (float)BS2d.y), ceil(ntheta / (float)BS2d.z));
  GS2d2 = dim3(ceil(n0 / (float)BS2d.x), ceil(n1 / (float)BS2d.y), ceil(ntheta / (float)BS2d.z));
 
}

// destructor, memory deallocation
holo::~holo() { free(); }

void holo::free() {
  if (!is_free) {
    cudaFree(fdee2d);
    cufftDestroy(plan2dchunk);
    is_free = true;
  }
}

void holo::fwd_usfft(size_t g_, size_t f_, size_t x_, size_t y_, size_t stream_) {

  f = (float2 *)f_;
  g = (float2 *)g_;
  x = (float *)x_;
  y = (float *)y_;
  stream = (cudaStream_t)stream_;    
  
  cufftSetStream(plan2dchunk, stream);
  cudaMemsetAsync(fdee2d, 0, ntheta * (2 * n1e + 2 * m1) * (2 * n0e + 2 * m0) * sizeof(float2),stream);
  
  divker2d<<<GS2d0, BS2d, 0,stream>>>(fdee2d, f, n0e, n1e, ntheta, m0, m1, mu0, mu1, 0);
  fftshiftc2d<<<GS2d1, BS2d, 0,stream>>>(fdee2d, (2 * n0e + 2 * m0), (2 * n1e + 2 * m1), ntheta);
  cufftExecC2C(plan2dchunk, (cufftComplex *)&fdee2d[m0 + m1 * (2 * n0e + 2 * m0)].x, (cufftComplex *)&fdee2d[m0 + m1 * (2 * n0e + 2 * m0)].x, CUFFT_FORWARD);
  fftshiftc2d<<<GS2d1, BS2d, 0,stream>>>(fdee2d, (2 * n0e + 2 * m0), (2 * n1e + 2 * m1), ntheta);
  wrap2d<<<GS2d1, BS2d, 0,stream>>>(fdee2d, n0e, n1e, ntheta, m0, m1, 0);
  gather2d<<<GS2d2, BS2d, 0,stream>>>(g, fdee2d, x, y, m0, m1, mu0, mu1, n0e, n1e, ntheta, n0, n1, 0);  
}

void holo::adj_usfft(size_t f_, size_t g_, size_t x_, size_t y_, size_t stream_) {

  f = (float2 *)f_;
  g = (float2 *)g_;
  x = (float *)x_;
  y = (float *)y_;
  stream = (cudaStream_t)stream_;    
  
  cufftSetStream(plan2dchunk, stream);

  cudaMemsetAsync(fdee2d, 0, ntheta * (2 * n1e + 2 * m1) * (2 * n0e + 2 * m0) * sizeof(float2),stream);
  gather2d<<<GS2d2, BS2d, 0,stream>>>(g, fdee2d, x, y, m0, m1, mu0, mu1, n0e, n1e,ntheta, n0, n1, 1);  
  wrap2d<<<GS2d1, BS2d, 0,stream>>>(fdee2d, n0e, n1e, ntheta, m0, m1, 1);
  fftshiftc2d<<<GS2d1, BS2d, 0,stream>>>(fdee2d, (2 * n0e + 2 * m0), (2 * n1e + 2 * m1), ntheta);
  cufftExecC2C(plan2dchunk, (cufftComplex *)&fdee2d[m0 + m1 * (2 * n0e + 2 * m0)].x, (cufftComplex *)&fdee2d[m0 + m1 * (2 * n0e + 2 * m0)].x, CUFFT_INVERSE);
  fftshiftc2d<<<GS2d1, BS2d, 0,stream>>>(fdee2d, (2 * n0e + 2 * m0), (2 * n1e + 2 * m1), ntheta);
  divker2d<<<GS2d0, BS2d, 0,stream>>>(fdee2d, f, n0e, n1e, ntheta, m0, m1, mu0, mu1, 1);
}

void holo::fwd_padsym(size_t g_, size_t f_, size_t pad_width, size_t ns, size_t stream_)
{
  f = (float2 *)f_;
  g = (float2 *)g_;
  stream = (cudaStream_t)stream_;    
  
  
  dim3 GS = dim3(ceil((n0+2*pad_width) / (float)BS2d.x), ceil((n1+2*pad_width) / (float)BS2d.y), ceil(ns / (float)BS2d.z));
  pad_sym <<<GS, BS2d, 0,stream>>> (g,f,pad_width,n0,n1,ns,0);  
}

void holo::adj_padsym(size_t g_, size_t f_, size_t pad_width, size_t ns,  size_t stream_)
{
  f = (float2 *)f_;
  g = (float2 *)g_;
  stream = (cudaStream_t)stream_;    
  
  dim3 GS = dim3(ceil((n0+2*pad_width) / (float)BS2d.x), ceil((n1+2*pad_width) / (float)BS2d.y), ceil(ns / (float)BS2d.z));
  pad_sym <<<GS, BS2d, 0,stream>>> (f,g,pad_width,n0,n1,ns,1);  
}