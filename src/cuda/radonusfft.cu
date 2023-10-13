#include <stdio.h>

#include "radonusfft.cuh"
#include "kernels_tomo.cu"
#include "shift.cu"
#include "filter.cu"

radonusfft::radonusfft(size_t ntheta, size_t pnz, size_t n, float center,
                       size_t theta_, size_t ngpus)
    : ntheta(ntheta), pnz(pnz), n(n), center(center), ngpus(ngpus) {
  float eps = 1e-2;
  mu = -log(eps) / (2 * n * n);
  m = ceil(2 * n * 1 / PI * sqrt(-mu * log(eps) + (mu * n) * (mu * n) / 4));
  f = new float2*[ngpus];
  g = new float2*[ngpus];
  fdee = new float2*[ngpus];
  x = new float*[ngpus];
  y = new float*[ngpus];
  shiftfwd = new float2*[ngpus];
  shiftadj = new float2*[ngpus];
  theta = new float*[ngpus];
  plan1d = new cufftHandle[ngpus];  
  plan2d = new cufftHandle[ngpus];
  
  for (int igpu=0;igpu<ngpus;igpu++)
  {
    cudaSetDevice(igpu);
    cudaMalloc((void **)&f[igpu], n * n * pnz * sizeof(float2));
    cudaMalloc((void **)&g[igpu], n * ntheta * pnz * sizeof(float2));
    cudaMalloc((void **)&fdee[igpu],
              (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));

    cudaMalloc((void **)&x[igpu], n * ntheta * sizeof(float));
    cudaMalloc((void **)&y[igpu], n * ntheta * sizeof(float));
    cudaMalloc((void **)&theta[igpu], ntheta * sizeof(float));
    cudaMemcpy(theta[igpu], (float *)theta_, ntheta * sizeof(float), cudaMemcpyDefault);
    
    int ffts[2];
    int idist;
    int inembed[2];
    // fft 2d
    ffts[0] = 2 * n;
    ffts[1] = 2 * n;
    idist = (2 * n + 2 * m) * (2 * n + 2 * m);
    inembed[0] = 2 * n + 2 * m;
    inembed[1] = 2 * n + 2 * m;
    cufftPlanMany(&plan2d[igpu], 2, ffts, inembed, 1, idist, inembed, 1, idist,
                  CUFFT_C2C, pnz);
    
    // fft 1d
    ffts[0] = n;
    idist = n;
    inembed[0] = n;
    cufftPlanMany(&plan1d[igpu], 1, ffts, inembed, 1, idist, inembed, 1, idist,
                  CUFFT_C2C, ntheta * pnz);
    cudaMalloc((void **)&shiftfwd[igpu], n * sizeof(float2));
    cudaMalloc((void **)&shiftadj[igpu], n * sizeof(float2));
    // compute shifts with respect to the rotation center
    takeshift <<<ceil(n / 1024.0), 1024>>> (shiftfwd[igpu], -(center - n / 2.0), n);
    takeshift <<<ceil(n / 1024.0), 1024>>> (shiftadj[igpu], (center - n / 2.0), n);


  }

  //back tp 0
  cudaSetDevice(0);


  BS2d = dim3(32, 32);
  BS3d = dim3(32, 32, 1);

  GS2d0 = dim3(ceil(n / (float)BS2d.x), ceil(ntheta / (float)BS2d.y));
  GS3d0 = dim3(ceil(n / (float)BS3d.x), ceil(n / (float)BS3d.y),
              ceil(pnz / (float)BS3d.z));
  GS3d1 = dim3(ceil(2 * n / (float)BS3d.x), ceil(2 * n / (float)BS3d.y),
              ceil(pnz / (float)BS3d.z));
  GS3d2 = dim3(ceil((2 * n + 2 * m) / (float)BS3d.x),
              ceil((2 * n + 2 * m) / (float)BS3d.y), ceil(pnz / (float)BS3d.z));
  GS3d3 = dim3(ceil(n / (float)BS3d.x), ceil(ntheta / (float)BS3d.y),
              ceil(pnz / (float)BS3d.z));
  
}

// destructor, memory deallocation
radonusfft::~radonusfft() { free(); }

void radonusfft::free() {
  if (!is_free) {
    for(int igpu=0;igpu<ngpus;igpu++)
    {
      cudaSetDevice(igpu);
      cudaFree(f[igpu]);
      cudaFree(g[igpu]);
      cudaFree(fdee[igpu]);
      cudaFree(x[igpu]);
      cudaFree(y[igpu]);
      cudaFree(shiftfwd[igpu]);
      cudaFree(shiftadj[igpu]);
      cufftDestroy(plan2d[igpu]);
      cufftDestroy(plan1d[igpu]);
    }
    cudaFree(f);
    cudaFree(g);
    cudaFree(fdee);
    cudaFree(x);
    cudaFree(y);
    cudaFree(shiftfwd);
    cudaFree(shiftadj);   
    is_free = true;   
  }
}

void radonusfft::fwd(size_t g_, size_t f_, size_t igpu) {

    cudaSetDevice(igpu);
    float2* f0 = (float2 *)f_;
    cudaMemcpy(f[igpu], f0, n * n * pnz * sizeof(float2), cudaMemcpyDefault);      
    cudaMemset(fdee[igpu], 0, (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));

    //circ <<<GS3d0, BS3d>>> (f, 1.0f / n, n, pnz);
    takexy <<<GS2d0, BS2d>>> (x[igpu], y[igpu], theta[igpu], n, ntheta);

    divphi <<<GS3d2, BS3d>>> (fdee[igpu], f[igpu], mu, n, pnz, m, TOMO_FWD);
    fftshiftc <<<GS3d2, BS3d>>> (fdee[igpu], 2 * n + 2 * m, pnz);
    cufftExecC2C(plan2d[igpu], (cufftComplex *)&fdee[igpu][m + m * (2 * n + 2 * m)],
                (cufftComplex *)&fdee[igpu][m + m * (2 * n + 2 * m)], CUFFT_FORWARD);
    fftshiftc <<<GS3d2, BS3d>>> (fdee[igpu], 2 * n + 2 * m, pnz);

    wrap <<<GS3d2, BS3d>>> (fdee[igpu], n, pnz, m, TOMO_FWD);
    gather <<<GS3d3, BS3d>>> (g[igpu], fdee[igpu], x[igpu], y[igpu], m, mu, n, ntheta, pnz, TOMO_FWD);
    // shift with respect to given center
    shift <<<GS3d3, BS3d>>> (g[igpu], shiftfwd[igpu], n, ntheta, pnz);

    ifftshiftc <<<GS3d3, BS3d>>> (g[igpu], n, ntheta, pnz);
    cufftExecC2C(plan1d[igpu], (cufftComplex *)g[igpu], (cufftComplex *)g[igpu], CUFFT_INVERSE);
    ifftshiftc <<<GS3d3, BS3d>>> (g[igpu], n, ntheta, pnz);
    if(n%4!=0)
      ifftshiftcmul <<<GS3d3, BS3d>>> (g[igpu], n, ntheta, pnz);

    float2* g0 = (float2 *)g_;
    for (int i=0;i<ntheta;i++)    
      cudaMemcpy(&g0[i*n*pnz], &g[igpu][i*n*pnz], n * pnz * sizeof(float2), cudaMemcpyDefault);  
}

void radonusfft::adj(size_t f_, size_t g_, size_t igpu, bool filter) {
    cudaSetDevice(igpu);
    float2* g0 = (float2 *)g_;
    for (int i=0;i<ntheta;i++)    
      cudaMemcpy(&g[igpu][i*n*pnz],&g0[i*n*pnz], n * pnz * sizeof(float2), cudaMemcpyDefault);
    cudaMemset(fdee[igpu], 0, (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));

    takexy <<<GS2d0, BS2d>>> (x[igpu], y[igpu], theta[igpu], n, ntheta);

    ifftshiftc <<<GS3d3, BS3d>>> (g[igpu], n, ntheta, pnz);
    cufftExecC2C(plan1d[igpu], (cufftComplex *)g[igpu], (cufftComplex *)g[igpu], CUFFT_FORWARD);
    ifftshiftc <<<GS3d3, BS3d>>> (g[igpu], n, ntheta, pnz);
    if(n%4!=0)
      ifftshiftcmul <<<GS3d3, BS3d>>> (g[igpu], n, ntheta, pnz);
  
    if (filter)
      applyfilter<<<GS3d3, BS3d>>>(g[igpu],n,ntheta,pnz);
    // shift with respect to given center
    shift <<<GS3d3, BS3d>>> (g[igpu], shiftadj[igpu], n, ntheta, pnz);

    gather <<<GS3d3, BS3d>>> (g[igpu], fdee[igpu], x[igpu], y[igpu], m, mu, n, ntheta, pnz, TOMO_ADJ);
    wrap <<<GS3d2, BS3d>>> (fdee[igpu], n, pnz, m, TOMO_ADJ);

    fftshiftc <<<GS3d2, BS3d>>> (fdee[igpu], 2 * n + 2 * m, pnz);
    cufftExecC2C(plan2d[igpu], (cufftComplex *)&fdee[igpu][m + m * (2 * n + 2 * m)],
                (cufftComplex *)&fdee[igpu][m + m * (2 * n + 2 * m)], CUFFT_INVERSE);
    fftshiftc <<<GS3d2, BS3d>>> (fdee[igpu], 2 * n + 2 * m, pnz);
    
    divphi <<<GS3d0, BS3d>>> (fdee[igpu], f[igpu], mu, n, pnz, m, TOMO_ADJ);
    //circ <<<GS3d0, BS3d>>> (f, 1.0f / n, n, pnz);
    float2* f0 = (float2 *)f_;
    cudaMemcpy(f0, f[igpu], n * n * pnz * sizeof(float2),
              cudaMemcpyDefault);
  //}
}
