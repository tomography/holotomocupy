%module holo

%{
#define SWIG_FILE_WITH_INIT
#include "holo.cuh"
%}

class holo 
{

public: 
  %mutable;  
  holo(size_t n0, size_t n1, size_t ntheta);
  ~holo();  
  void fwd_usfft(size_t g_, size_t f_, size_t x_, size_t y_, size_t stream_);
  void adj_usfft(size_t g_, size_t f_, size_t x_, size_t y_,  size_t stream_);
  void fwd_padsym(size_t g_, size_t f_, size_t pad_width,  size_t stream_);
  void adj_padsym(size_t g_, size_t f_, size_t pad_width,  size_t stream_);
  
  void free();
};
