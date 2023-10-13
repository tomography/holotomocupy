%module holotomo

%{
#define SWIG_FILE_WITH_INIT
#include "holotomo.cuh"
%}

class holotomo 
{

public: 
  %mutable;  
  holotomo(size_t n0, size_t n1, size_t ntheta);
  ~holotomo();  
  void fwd_usfft(size_t g_, size_t f_, size_t x_, size_t y_, size_t stream_);
  void adj_usfft(size_t g_, size_t f_, size_t x_, size_t y_,  size_t stream_);
  void free();
};
