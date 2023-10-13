#define PI 3.1415926535
// Pytchography kernels

void __global__ mul(float2 *g, float2 *f, float2 *prb, float *scanx, float *scany,
	int Ntheta, int Nz, int N, int Nscan, int Nprb, int detx, int dety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=Nprb*Nprb||ty>=Nscan||tz>=Ntheta) return;
	int iy = tx/Nprb;
	int ix = tx%Nprb;

	int stx = roundf(scanx[ty+tz*Nscan]);
	int sty = roundf(scany[ty+tz*Nscan]);
	if(stx<0||sty<0||stx>N-1||sty>Nz-1) return;

	int shift = (dety-Nprb)/2*detx+(detx-Nprb)/2;
	float2 f0 = f[(stx+ix)+(sty+iy)*N+tz*Nz*N];
	float2 prb0 = prb[ix+iy*Nprb+tz*Nprb*Nprb];
	float c = 1/sqrtf(detx*dety);//fft constant
	g[shift+ix+iy*detx+ty*detx*dety+tz*detx*dety*Nscan].x = c*prb0.x*f0.x-c*prb0.y*f0.y;
	g[shift+ix+iy*detx+ty*detx*dety+tz*detx*dety*Nscan].y = c*prb0.x*f0.y+c*prb0.y*f0.x;

}

void __global__ mula(float2 *f, float2 *g, float2 *prb, float *scanx, float *scany,
	int Ntheta, int Nz, int N, int Nscan, int Nprb, int detx, int dety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=Nprb*Nprb||ty>=Nscan||tz>=Ntheta) return;
	int iy = tx/Nprb;
	int ix = tx%Nprb;

	int stx = roundf(scanx[ty+tz*Nscan]);
	int sty = roundf(scany[ty+tz*Nscan]);
	if(stx<0||sty<0||stx>N-1||sty>Nz-1) return;

	int shift = (dety-Nprb)/2*detx+(detx-Nprb)/2;
	float2 g0 = g[shift+ix+iy*detx+ty*detx*dety+tz*detx*dety*Nscan];
	float2 prb0 = prb[ix+iy*Nprb+tz*Nprb*Nprb];
	float c = 1/sqrtf(detx*dety);//fft constant
	atomicAdd(&f[(stx+ix)+(sty+iy)*N+tz*Nz*N].x, c*prb0.x*g0.x+c*prb0.y*g0.y);
	atomicAdd(&f[(stx+ix)+(sty+iy)*N+tz*Nz*N].y, c*prb0.x*g0.y-c*prb0.y*g0.x);
}

void __global__ mulaprb(float2 *f, float2 *g, float2 *prb, float *scanx, float *scany,
	int Ntheta, int Nz, int N, int Nscan, int Nprb, int detx, int dety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=Nprb*Nprb||ty>=Nscan||tz>=Ntheta) return;
	int iy = tx/Nprb;
	int ix = tx%Nprb;

	int stx = roundf(scanx[ty+tz*Nscan]);
	int sty = roundf(scany[ty+tz*Nscan]);
	if(stx<0||sty<0) return;

	int shift = (dety-Nprb)/2*detx+(detx-Nprb)/2;
	
	float2 g0 = g[shift+ix+iy*detx+ty*detx*dety+tz*detx*dety*Nscan];
	float2 f0 = f[(stx+ix)+(sty+iy)*N+tz*Nz*N];
	float c = 1/sqrtf(detx*dety);//fft constant
	atomicAdd(&prb[ix+iy*Nprb+tz*Nprb*Nprb].x, c*f0.x*g0.x+c*f0.y*g0.y);
	atomicAdd(&prb[ix+iy*Nprb+tz*Nprb*Nprb].y, c*f0.x*g0.y-c*f0.y*g0.x);
}

void __global__ updatepsi(float2* f, float2* ff, float2* ftmp0, float2* ftmp1,
	float2* fff, float rho, float gamma, float maxint, int Ntheta, int Nz,int N)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=N||ty>=Nz||tz>=Ntheta) return;

	int ind = tx+ty*N+tz*N*Nz;
	f[ind].x = (1-rho*gamma)*f[ind].x+rho*gamma*(ff[ind].x-fff[ind].x/rho) +
				gamma/2*(ftmp0[ind].x-ftmp1[ind].x)/maxint;
	f[ind].y = (1-rho*gamma)*f[ind].y+rho*gamma*(ff[ind].y-fff[ind].y/rho) +
				gamma/2*(ftmp0[ind].y-ftmp1[ind].y)/maxint;
}

void __global__ takeshifts(float2* shiftx,float2* shifty,float* scanx,float* scany,int Ntheta, int Nscan)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tx>=Nscan||ty>=Ntheta) return;
	int ind = tx+ty*Nscan;
	shiftx[ind].x = cosf(2*PI*(scanx[ind] - roundf(scanx[ind])));
	shiftx[ind].y = sinf(2*PI*(scanx[ind] - roundf(scanx[ind])));
	shifty[ind].x = cosf(2*PI*(scany[ind] - roundf(scany[ind])));
	shifty[ind].y = sinf(2*PI*(scany[ind] - roundf(scany[ind])));
}

void __global__ shifts(float2* f, float2* shiftx,float2* shifty,int Ntheta, int Nscan, int detxdety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=detxdety||ty>=Nscan||tz>=Ntheta) return;
	int ind = tx+ty*detxdety+tz*detxdety*Nscan;
	int inds = ty+tz*Nscan;
	float2 f0 = f[ind];
	float2 shiftx0 = shiftx[inds];
	float2 shifty0 = shifty[inds];
	f[ind].x = f0.x*shiftx0.x-f0.y*shiftx0.y;
	f[ind].y = f0.y*shiftx0.x+f0.x*shiftx0.y;
	f0 = f[ind];
	f[ind].x = f0.x*shifty0.x-f0.y*shifty0.y;
	f[ind].y = f0.y*shifty0.x+f0.x*shifty0.y;
}

void __global__ shiftsa(float2* f, float2* shiftx,float2* shifty,int Ntheta, int Nscan, int detxdety)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;

	if (tx>=detxdety||ty>=Nscan||tz>=Ntheta) return;
	int ind = tx+ty*detxdety+tz*detxdety*Nscan;
	int inds = ty+tz*Nscan;
	float2 f0 = f[ind];
	float2 shiftx0 = shiftx[inds];
	float2 shifty0 = shifty[inds];
	f[ind].x = f0.x*shiftx0.x+f0.y*shiftx0.y;
	f[ind].y = f0.y*shiftx0.x-f0.x*shiftx0.y;
	f0 = f[ind];
	f[ind].x = f0.x*shifty0.x+f0.y*shifty0.y;
	f[ind].y = f0.y*shifty0.x-f0.x*shifty0.y;
}
