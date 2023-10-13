#ifndef fft_CUH
#define fft_CUH

#include <cufft.h>


class holotomo {
  bool is_free = false;
  
  
  float2 *f;
  float2 *g;
  
  float2 *fdee2d;
  float* x;
  float* y;
  cufftHandle plan2dchunk;
  cudaStream_t stream;
  
  dim3 BS2d, GS2d0, GS2d1, GS2d2;
  
  size_t n0,n1,ntheta;  
  size_t m0,m1;
  float mu0;float mu1;
public:  
  holotomo(size_t n0, size_t n1, size_t ntheta);  
  ~holotomo();  
  void fwd_usfft(size_t g_, size_t f_, size_t x_, size_t y_, size_t stream_);
  void adj_usfft(size_t f_, size_t g_, size_t x_, size_t y_,  size_t stream_);
  void free();
};

#endif