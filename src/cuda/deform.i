/*interface*/
%module deform

%{
#define SWIG_FILE_WITH_INIT
#include "deform.cuh"
%}

class deform
{

public:
  %immutable;
  size_t n;
  size_t ntheta;
  size_t nz; 
  size_t ptheta;
  size_t ngpus;
  %mutable;
  deform(size_t ntheta,size_t nz, size_t n, size_t ptheta, size_t ngpus);
  ~deform();  
  void free();
  void remap(size_t g, size_t f, size_t flowx, size_t flowy, size_t igpu);

};
