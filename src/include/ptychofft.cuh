#include <cufft.h>

class ptychofft
{
	bool is_free = false;
	float2** shiftx; 
	float2** shifty; 
	
	cufftHandle* plan2d;
public:
	size_t n;
	size_t ptheta;
	size_t nz;
	size_t nscan;
	size_t ndet;
	size_t nprb;
	size_t ngpus;
	
	ptychofft(size_t ptheta, size_t nz, size_t n,
		size_t nscan, size_t ndet, size_t nprb, size_t ngpus);
	~ptychofft();
	void fwd(size_t g, size_t f, size_t prb, size_t scan, size_t igpu);
	void adj(size_t f, size_t g, size_t prb, size_t scan, size_t igpu);	
	void adjprb(size_t prb, size_t g, size_t scan, size_t f, size_t igpu);
	void free();
};
