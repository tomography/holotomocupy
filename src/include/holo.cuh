#ifndef fft_CUH
#define fft_CUH

#include <cufft.h>


class holo {
  bool is_free = false;
  
  
  float2 *f;
  float2 *g;
  
  float2 *fdee2d;
  float* x;
  float* y;
  cufftHandle plan2dchunk;
  cudaStream_t stream;
  
  dim3 BS2d, GS2d0, GS2d1, GS2d2;
  
  size_t n0,n1,n0e,n1e,ntheta;  
  size_t m0,m1;
  float mu0;float mu1;
public:  
  holo(size_t n0e, size_t n1e,size_t n0, size_t n1, size_t ntheta);
  ~holo();  
  void fwd_usfft(size_t g_, size_t f_, size_t x_, size_t y_, size_t stream_);
  void adj_usfft(size_t f_, size_t g_, size_t x_, size_t y_,  size_t stream_);
  void fwd_padsym(size_t g_, size_t f_, size_t pad_width,  size_t ns, size_t stream_);
  void adj_padsym(size_t g_, size_t f_, size_t pad_width,  size_t ns, size_t stream_);
  
  void free();
};

#endif
