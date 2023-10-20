/*interface*/
%module tomo

%{
#define SWIG_FILE_WITH_INIT
#include "tomo.cuh"
%}

class tomo
{
public:
  %immutable;
  size_t n;
  size_t ntheta;
  size_t pnz;
  float center;
  size_t ngpus;

  %mutable;
  tomo(size_t ntheta, size_t pnz, size_t n, float center, size_t theta_, size_t ngpus);
  ~tomo();
  void fwd(size_t g, size_t f, size_t igpu);
  void adj(size_t f, size_t g, size_t igpu, bool filter);
  void free();
};
