void __global__ applyfilter(float2 *f, int N, int Ntheta, int Nz) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= N || ty >= Ntheta || tz >= Nz)
    return;
  int id0 = tx + ty * N + tz * Ntheta * N;
  float rho = (tx - N / 2) / (float)N;
  float w = 0;
  float coef=0.5;
  if (rho == 0)
    w = 0;
  else
    w = abs(rho) * N * 4 * (1-fabs(rho)/coef)*(1-fabs(rho)/coef)*(1-fabs(rho)/coef);
  f[id0].x *= w;
  f[id0].y *= w;
}
