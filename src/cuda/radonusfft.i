/*interface*/
%module radonusfft

%{
#define SWIG_FILE_WITH_INIT
#include "radonusfft.cuh"
%}

class radonusfft
{
public:
  %immutable;
  size_t n;
  size_t ntheta;
  size_t pnz;
  float center;
  size_t ngpus;

  %mutable;
  radonusfft(size_t ntheta, size_t pnz, size_t n, float center, size_t theta_, size_t ngpus);
  ~radonusfft();
  void fwd(size_t g, size_t f, size_t igpu);
  void adj(size_t f, size_t g, size_t igpu, bool filter);
  void free();
};
