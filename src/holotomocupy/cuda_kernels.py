import cupy as cp

gather_kernel = cp.RawKernel(
    r"""
// Forward only: R gathers polar samples out of the padded Cartesian spectrum.
// The adjoint is `scatter_binned` below, which must stay consistent with every
// geometry decision made here -- the guard, the d_ax phase and the stencil.
extern "C" __global__ void gather(float2* g, float2* f, float* theta, int m, float* mu,
                                  int n, int ndet, int ntheta, int nz)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= ndet || ty >= ntheta || tz >= nz) return;

    const float PI     = 3.141592653589793238f;
    const int   twon   = 2 * n;
    const float ftwon  = (float)twon;
    const float mu0    = mu[0];
    const float coeff0 = PI / mu0;
    const float coeff1 = -PI * PI / mu0;
    const float inv_twon = 1.0f / ftwon;

    const int g_ind = tx + tz * ndet + ty * ndet * nz;  // swapped axes

    // Detector frequency in cycles per *object* pixel.  The detector spans the
    // same field of view as the object (ndet samples of size n/ndet), so the
    // frequency step is 1/n whatever ndet is -- only the range changes, to
    // |fr| <= ndet/(2n).  For ndet == n this is the usual (tx - n/2)/n.
    const float cx = ndet * 0.5f;
    const float fr = (tx - cx) / (float)n;

    // Samples outside the padded FFT square are skipped, not wrapped; they only
    // exist once ndet > n.  Wrapping aliased them, which at theta = 0 and 90 is
    // exact rather than approximate and made those two rows combs -- the source
    // of the line artifacts on the tomo_upsample=2 path.  The test is on
    // (x0, y0), not |fr|: the representable spectrum is a square, so a radial
    // line reaches sqrt(2)/2 at 45 deg and guarding on |fr| would drop corners.
    // The % twon in the loops below is a different thing and stays: that is the
    // stencil straddling the edge of a genuinely periodic spectrum.

    const float x0 =  fr * __cosf(theta[ty]);
    const float y0 = -fr * __sinf(theta[ty]);

    // Strict > on the upper side: +1/2 and -1/2 are the same frequency here, so
    // the fr = -1/2 bin is legitimately wrapped.  It is the tx = 0 bin of every
    // projection at ndet == n, where the guard must never fire.
    if (x0 < -0.5f || x0 > 0.5f || y0 < -0.5f || y0 > 0.5f)
    {
        g[g_ind] = make_float2(0.0f, 0.0f);
        return;
    }

    // Put the rotation axis at (n-1)/2, not n/2, so it stays at the detector
    // middle under binning instead of drifting by 0.5*(scale-1).  Undoing the
    // difference is one phase, exp(-2i*pi*fr*d_ax), with d_ax in object pixels:
    // half an object pixel for the object axis, half a DETECTOR pixel (= n/ndet
    // object px) for the detector origin.  The two coincide at ndet == n, which
    // is why a flat 0.5f was only ever wrong on the tomo_upsample=2 path.
    // `scatter_binned` applies the conjugate, keeping R and RT adjoint.
    const float d_ax = 0.5f * (__cosf(theta[ty]) - __sinf(theta[ty]))
                     - 0.5f * (float)n / (float)ndet;
    float sn_ax, cs_ax;
    __sincosf(-6.283185307179586f * fr * d_ax, &sn_ax, &cs_ax);

    float2 g0 = make_float2(0.0f, 0.0f);

    const int base_x  = (int)floorf(ftwon * x0) - m;
    const int base_y  = (int)floorf(ftwon * y0) - m;
    const int tz_off  = tz * twon * twon;
    const int len     = 2 * m + 1;

    // Precompute x-direction exponential factors once.
    // Reduces expf calls from (2m+1)^2 to 2*(2m+1).
    float ex[32];  // 2*m+1 entries; m is small (typically 4-5)
    for (int i0 = 0; i0 < len; i0++) {
        float w0 = (base_x + i0) * inv_twon - x0;
        ex[i0] = __expf(coeff1 * w0 * w0);
    }

    for (int i1 = 0; i1 < len; i1++)
    {
        int   ell1    = base_y + i1;
        float w1      = ell1 * inv_twon - y0;
        float ey      = coeff0 * __expf(coeff1 * w1 * w1);
        int   f_indy  = (n + ell1 + twon) % twon;
        int   row_off = twon * f_indy + tz_off;

        for (int i0 = 0; i0 < len; i0++)
        {
            float w    = ex[i0] * ey;
            int   ell0 = base_x + i0;
            int   f_ind = (n + ell0 + twon) % twon + row_off;

            g0.x += w * f[f_ind].x;
            g0.y += w * f[f_ind].y;
        }
    }

    const float gx = cs_ax * g0.x - sn_ax * g0.y;          // phase * g0
    const float gy = sn_ax * g0.x + cs_ax * g0.y;
    g[g_ind].x = gx / n;
    g[g_ind].y = gy / n;
}
""",
    "gather",
)

# ---------------------------------------------------------------------------
# Binned scatter: RT, the adjoint of `gather` above.  Pushing each sample into
# global memory costs (2m+1)^2 atomicAdds -- 4.6 billion at n = 2048.  Instead
# `sample_tiles` lists once which samples reach which b x b tile, and
# `scatter_binned` gives each tile to a block that accumulates in shared memory
# and writes global once.  Measured 2.5-3.1x faster over n = 256..4736.
#
# Both must agree with `gather` on fr, the out-of-square guard and the d_ax
# phase; those are copied from it verbatim.
# ---------------------------------------------------------------------------

sample_tiles_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void sample_tiles(int* out, const float* theta, int m,
                                        int n, int ndet, int a0, int na,
                                        int b, int ntx)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;   // detector bin
    int ty = blockDim.y * blockIdx.y + threadIdx.y;   // angle, relative to a0
    if (tx >= ndet || ty >= na) return;

    const int   twon  = 2 * n;
    const float ftwon = (float)twon;

    const float fr = (tx - ndet * 0.5f) / (float)n;
    const float x0 =  fr * __cosf(theta[a0 + ty]);
    const float y0 = -fr * __sinf(theta[a0 + ty]);

    const int o = 4 * (ty * ndet + tx);

    // The gather kernel's out-of-square guard.  Those samples read back as zero
    // and scatter nothing, so they belong to no tile and are simply left out of
    // the index -- which is also what keeps the two directions adjoint.
    if (x0 < -0.5f || x0 > 0.5f || y0 < -0.5f || y0 > 0.5f) {
        out[o + 0] = -1; out[o + 1] = -1; out[o + 2] = -1; out[o + 3] = -1;
        return;
    }

    int px = n + (int)floorf(ftwon * x0) - m;  if (px < 0) px += twon;
    int py = n + (int)floorf(ftwon * y0) - m;  if (py < 0) py += twon;

    int ex = px + 2 * m;  if (ex >= twon) ex -= twon;
    int ey = py + 2 * m;  if (ey >= twon) ey -= twon;

    // At most two tiles per axis, so four entries suffice.  The host picks b so
    // that every tile, including a partial last one, is at least 2m+1 wide --
    // a narrower one could sit entirely inside a stencil and be missed here.
    int ix0 = px / b, ix1 = ex / b;
    int iy0 = py / b, iy1 = ey / b;

    out[o + 0] = iy0 * ntx + ix0;
    out[o + 1] = (ix1 != ix0)               ? (iy0 * ntx + ix1) : -1;
    out[o + 2] = (iy1 != iy0)               ? (iy1 * ntx + ix0) : -1;
    out[o + 3] = (ix1 != ix0 && iy1 != iy0) ? (iy1 * ntx + ix1) : -1;
}
""",
    "sample_tiles",
)


scatter_binned_kernel = cp.RawKernel(
    r"""
#define CH 256                     // samples staged per pass == blockDim.x
extern "C" __global__ void scatter_binned(
    float2* __restrict__ f, const float2* __restrict__ g,
    const float* __restrict__ theta,
    const int* __restrict__ bin_samples,
    const int* __restrict__ sub_tile,
    const int* __restrict__ sub_beg,
    const int* __restrict__ sub_end,
    const unsigned char* __restrict__ sub_atomic,
    int m, const float* __restrict__ mu,
    int n, int ndet, int ntheta, int nz, int b, int ntx)
{
    // Real and imaginary as two planes with row stride b+1, not interleaved
    // float2: interleaved, a warp's .x writes hit only 16 of the 32 banks and
    // serialise.  With this layout the shared atomics cost nothing measurable
    // (143.7 ms against 144.1 ms for plain racy adds at n = 3072).
    extern __shared__ float sh[];          // 2 * b * (b+1) floats
    __shared__ int    ss[CH];              // staged sample indices
    __shared__ float2 sg[CH];              // staged sinogram values
    const int sub = blockIdx.x;
    const int tz  = blockIdx.y;
    const int nt  = b * (b + 1);
    float* const shx = sh;
    float* const shy = sh + nt;

    for (int i = threadIdx.x; i < 2 * nt; i += blockDim.x) sh[i] = 0.0f;

    const int tile = sub_tile[sub];
    const int tx0  = (tile % ntx) * b;
    const int ty0  = (tile / ntx) * b;

    const float PI       = 3.141592653589793238f;
    const int   twon     = 2 * n;
    // ceil(2n/b) tiles per axis, so a last row or column may hang off the grid;
    // bw/bh are this tile's real extent and everything below clips to them.
    const int   bw       = min(b, twon - tx0);
    const int   bh       = min(b, twon - ty0);
    const float ftwon    = (float)twon;
    const float mu0      = mu[0];
    const float coeff0   = PI / mu0;
    const float coeff1   = -PI * PI / mu0;
    const float inv_twon = 1.0f / ftwon;
    const int   len      = 2 * m + 1;

    const int beg = sub_beg[sub], end = sub_end[sub];

    // Consecutive entries are one angle's adjacent detector bins, 2 grid cells
    // apart, so giving them to adjacent lanes recreates the contention binning
    // exists to remove.  Stage coalesced (lane j reads entry j), then process
    // entry perm(j), which spreads a warp 16 cells apart -- wider than the
    // stencil, so no two lanes in a warp touch the same cell.
    const int jperm = (threadIdx.x & 31) * (CH / 32) + (threadIdx.x >> 5);

    for (int base = beg; base < end; base += CH)
    {
      const int cnt = min(CH, end - base);
      __syncthreads();
      if (threadIdx.x < cnt) {
          int s0 = bin_samples[base + threadIdx.x];
          int a0i = s0 / ndet;
          ss[threadIdx.x] = s0;
          sg[threadIdx.x] = g[(s0 - a0i * ndet) + tz * ndet + a0i * ndet * nz];
      }
      __syncthreads();
      if (jperm < cnt)
      {
        const int s = ss[jperm];
        const int a = s / ndet;            // angle
        const int c = s - a * ndet;        // detector bin

        const float fr  = (c - ndet * 0.5f) / (float)n;
        const float cs_t = __cosf(theta[a]), sn_t = __sinf(theta[a]);
        const float x0 =  fr * cs_t;
        const float y0 = -fr * sn_t;
        // No out-of-square guard here: sample_tiles already dropped those.

        // The gather kernel's rotation-axis phase, conjugated: this is the
        // adjoint of the phase R applies after gathering.
        const float d_ax = 0.5f * (cs_t - sn_t) - 0.5f * (float)n / (float)ndet;
        float sn_ax, cs_ax;
        __sincosf(-6.283185307179586f * fr * d_ax, &sn_ax, &cs_ax);
        const float2 gin = sg[jperm];
        const float2 g0  = make_float2( cs_ax * gin.x + sn_ax * gin.y,
                                       -sn_ax * gin.x + cs_ax * gin.y);

        const int base_x = (int)floorf(ftwon * x0) - m;
        const int base_y = (int)floorf(ftwon * y0) - m;
        int px = n + base_x;  if (px < 0) px += twon;
        int py = n + base_y;  if (py < 0) py += twon;

        // The part of the stencil in this tile is a contiguous run of i0, so a
        // first index and an offset suffice -- no (index, weight) list.  With
        // offx = tx0 - px it is [0, len) clamped to [offx, offx + bw); if that
        // is empty the stencil reaches the tile by wrapping, which shifts offx
        // by twon.  Both cases non-empty would need a stencil wider than 2n.
        int offx = tx0 - px;
        int ia = max(0, offx), ib = min(len, offx + bw);
        if (ib <= ia) { offx += twon; ia = max(0, offx); ib = min(len, offx + bw); }
        if (ib <= ia) goto done;

        int offy = ty0 - py;
        int ja = max(0, offy), jb = min(len, offy + bh);
        if (jb <= ja) { offy += twon; ja = max(0, offy); jb = min(len, offy + bh); }
        if (jb <= ja) goto done;

        float wxs[32];                     // 2m+1 entries, as gather's ex[32]
        for (int i0 = ia; i0 < ib; i0++) {
            float w0 = (base_x + i0) * inv_twon - x0;
            wxs[i0 - ia] = __expf(coeff1 * w0 * w0);
        }

        for (int i1 = ja; i1 < jb; i1++) {
            float w1  = (base_y + i1) * inv_twon - y0;
            float ey  = coeff0 * __expf(coeff1 * w1 * w1);
            int   row = (i1 - offy) * (b + 1) - offx;
            for (int i0 = ia; i0 < ib; i0++) {
                float w = wxs[i0 - ia] * ey;
                int   o = row + i0;
                atomicAdd(&shx[o], w * g0.x);
                atomicAdd(&shy[o], w * g0.y);
            }
        }
      }
      done: ;
    }
    __syncthreads();

    // A tile small enough to be one subproblem is owned outright by this
    // block, so it can be stored rather than accumulated.  Split tiles have to
    // combine, but at b*b global atomics per subproblem instead of 81 per
    // sample -- and most of a split tile's cells are usually still zero.
    float2* ftile = f + (size_t)tz * twon * twon;
    const bool at = sub_atomic[sub];
    // bw x bh, not b x b: the overhang of a last-row or last-column tile has
    // no cells in f and nothing ever accumulated into it.
    for (int c0 = threadIdx.x; c0 < bw * bh; c0 += blockDim.x) {
        int   lx = c0 % bw, ly = c0 / bw;
        int   i  = ly * (b + 1) + lx;
        float vx = shx[i], vy = shy[i];
        size_t o = (size_t)(ty0 + ly) * twon + (tx0 + lx);
        if (at) {
            if (vx != 0.0f) atomicAdd(&ftile[o].x, vx);
            if (vy != 0.0f) atomicAdd(&ftile[o].y, vy);
        } else {
            ftile[o] = make_float2(vx, vy);
        }
    }
}
""",
    "scatter_binned",
)

pad_fwd_kernel = cp.RawKernel(
    r"""
extern "C" void __global__ pad_fwd(float2* __restrict__ g,
                                    const float2* __restrict__ f,
                                    int n, int nz, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= 2*n || ty >= 2*nz || tz >= ntheta) return;

    int txx = (tx < n/2)       ? (n/2  - tx - 1)         :
              (tx >= n + n/2)   ? (2*n  - tx + n/2  - 1)  : (tx - n/2);
    int tyy = (ty < nz/2)      ? (nz/2 - ty - 1)         :
              (ty >= nz + nz/2) ? (2*nz - ty + nz/2 - 1)  : (ty - nz/2);

    g[tz*2*n*2*nz + ty*2*n + tx] = f[tz*n*nz + tyy*n + txx];
}
""",
    "pad_fwd",
)

pad_adj_kernel = cp.RawKernel(
    r"""
/* Adjoint of pad_fwd: launch over f (n x nz).
   Each f[tx,ty] gathers from exactly 4 symmetric locations in g — no atomics. */
extern "C" void __global__ pad_adj(const float2* __restrict__ g,
                                    float2* __restrict__ f,
                                    int n, int nz, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;
    if (tx >= n || ty >= nz || tz >= ntheta) return;

    int gx_c = tx + n/2;
    int gx_m = (tx < n/2) ? (n/2 - 1 - tx) : (2*n + n/2 - 1 - tx);
    int gy_c = ty + nz/2;
    int gy_m = (ty < nz/2) ? (nz/2 - 1 - ty) : (2*nz + nz/2 - 1 - ty);

    const float2* base = g + tz * 2*n * 2*nz;
    float2 v0 = base[gy_c*2*n + gx_c];
    float2 v1 = base[gy_c*2*n + gx_m];
    float2 v2 = base[gy_m*2*n + gx_c];
    float2 v3 = base[gy_m*2*n + gx_m];
    f[tz*n*nz + ty*n + tx] = {v0.x+v1.x+v2.x+v3.x, v0.y+v1.y+v2.y+v3.y};
}
""",
    "pad_adj",
)

# B-spline basis functions and derivatives.
# Use fabsf instead of an integer sgn variable to avoid branching.
fun_phi = r"""
__device__ __forceinline__ float phi(float t)
{
    if (-2.0f < t && t <= -1.0f) return (t + 2.0f) * (t + 2.0f) * (t + 2.0f);
    if (-1.0f < t && t <=  1.0f) return 4.0f - 6.0f*t*t + 3.0f*fabsf(t)*t*t;
    if ( 1.0f < t && t <=  2.0f) return (2.0f - t) * (2.0f - t) * (2.0f - t);
    return 0.0f;
}
__device__ __forceinline__ int sym_idx(int i, int N)
{
    if (i < 0)   i = -i;
    if (i >= N)  i = 2*N - 2 - i;
    return i;
}
"""

fun_dphi = r"""
__device__ __forceinline__ float dphi(float t)
{
    if (-2.0f < t && t <= -1.0f) return 3.0f * (t + 2.0f) * (t + 2.0f);
    if (-1.0f < t && t <=  1.0f) return -12.0f*t + 9.0f*fabsf(t)*t;
    if ( 1.0f < t && t <=  2.0f) return -3.0f * (2.0f - t) * (2.0f - t);
    return 0.0f;
}
"""

fun_d2phi = r"""
__device__ __forceinline__ float d2phi(float t)
{
    if (-2.0f < t && t <= -1.0f) return 6.0f * (t + 2.0f);
    if (-1.0f < t && t <=  1.0f) return -12.0f + 18.0f*fabsf(t);
    if ( 1.0f < t && t <=  2.0f) return 6.0f * (2.0f - t);
    return 0.0f;
}
"""

s_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + r"""
void __global__ s(float2* g, float2* f, float* r, float* mag,
                  int n, int npsi, int nz, int nzpsi, int ntheta, bool dir)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x      = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y      = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   g_ind  = tx + ty * n + tz * n * nz;
    const int   tz_off = tz * npsi * nzpsi;

    // Precompute x-direction phi values once (4 evals instead of 16).
    float px[4];
    for (int jx = -1; jx < 3; jx++) px[jx + 1] = phi(dx - jx);

    float2 g0 = (dir == 0) ? make_float2(0.0f, 0.0f) : g[g_ind];

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        int indy_s = sym_idx(indy, nzpsi);
        float pdym    = phi(dy - jy);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w   = px[jx + 1] * pdym;
            int   idx = indx_s + row_off;

            if (dir == 0)
            {
                g0.x += w * f[idx].x;
                g0.y += w * f[idx].y;
            }
            else
            {
                atomicAdd(&(f[idx].x), w * g0.x);
                atomicAdd(&(f[idx].y), w * g0.y);
            }
        }
    }

    if (dir == 0) g[g_ind] = g0;
}
}
""",
    "s",
)


# extra for paganin

fun_phi_back = r"""
__device__ __forceinline__ float phi(float t, float m)
{
    t /= m;
    if (-2.0f < t && t <= -1.0f) return (t + 2.0f) * (t + 2.0f) * (t + 2.0f);
    if (-1.0f < t && t <=  1.0f) return 4.0f - 6.0f*t*t + 3.0f*fabsf(t)*t*t;
    if ( 1.0f < t && t <=  2.0f) return (2.0f - t) * (2.0f - t) * (2.0f - t);
    return 0.0f;
}
"""

sback_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + r"""
void __global__ sback(float2* g, float2* f, float* r, float* mag,
                      int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;  // in [0, npsi)
    int ty = blockDim.y * blockIdx.y + threadIdx.y;  // in [0, nzpsi)
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= npsi || ty >= nzpsi || tz >= ntheta) return;

    const float x      = (tx - (npsi-1) * 0.5f + r[2 * tz + 1]) / mag[2 * tz + 1] + (n-1)   * 0.5f;
    const float y      = (ty - (nzpsi-1)* 0.5f + r[2 * tz + 0]) / mag[2 * tz + 0] + (nz-1)  * 0.5f;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   g_ind  = tx + ty * npsi + tz * npsi * nzpsi;
    const int   tz_off = tz * n * nz;

    float px[4];
    for (int jx = -1; jx < 3; jx++) px[jx + 1] = phi(dx - jx);

    float2 g0 = make_float2(0.0f, 0.0f);

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        if (indy < 0 || indy >= nz) continue;
        float pdym    = phi(dy - jy);
        int   row_off = indy * n + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            if (indx < 0 || indx >= n) continue;
            float w   = px[jx + 1] * pdym;
            int   idx = indx + row_off;
            g0.x += w * f[idx].x;
            g0.y += w * f[idx].y;
        }
    }

    g[g_ind] = g0;
}
}
""",
    "sback",
)








d2s_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + fun_d2phi
    + r"""
// SLOT PAIRING: w1 is built from Deltar1 and multiplies c1, w2 from Deltar2
// and multiplies c2 -- each coefficient is contracted with the shift in its
// OWN slot. The mixed second differential needs the opposite, so the CALLER
// passes the coefficients crossed (see Shift.d2curlySc and Rec.d2F_dF3). The
// unfused reference splits this into dT(c1, dr2) + dT(c2, dr1) + d2T(c), where
// the crossing is visible in the Python instead.
void __global__ d2s(float2* res, float2* c, float2* c1, float2* c2, float* r, float* mag,
                    float* Deltar1, float* Deltar2,
                    int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x        = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y        = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix       = (int)floorf(x);
    const int   iy       = (int)floorf(y);
    const float dx       = x - ix;
    const float dy       = y - iy;
    const float Deltar1x = Deltar1[2 * tz + 1];
    const float Deltar1y = Deltar1[2 * tz + 0];
    const float Deltar2x = Deltar2[2 * tz + 1];
    const float Deltar2y = Deltar2[2 * tz + 0];
    const float cross    = Deltar1x * Deltar2y + Deltar1y * Deltar2x;
    const int   tz_off   = tz * npsi * nzpsi;

    // Precompute x-direction phi, dphi, d2phi values (12 evals instead of 48).
    float px[4], dpx[4], d2px[4];
    for (int jx = -1; jx < 3; jx++) {
        float d   = dx - jx;
        px[jx + 1]   = phi(d);
        dpx[jx + 1]  = dphi(d);
        d2px[jx + 1] = d2phi(d);
    }

    float2 r0 = {};

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        float d2pdym  = d2phi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w  = d2px[jx + 1] * pdym    * Deltar1x * Deltar2x
                     + dpx[jx + 1]  * dpdym   * cross
                     + px[jx + 1]   * d2pdym  * Deltar1y * Deltar2y;
            float w1 = dpx[jx + 1] * pdym  * Deltar1x
                     + dpdym        * px[jx + 1] * Deltar1y;
            float w2 = dpx[jx + 1] * pdym  * Deltar2x
                     + dpdym        * px[jx + 1] * Deltar2y;
            int idx = indx_s + row_off;
            r0.x += w  * c[idx].x;
            r0.y += w  * c[idx].y;
            r0.x -= w1 * c1[idx].x;
            r0.y -= w1 * c1[idx].y;
            r0.x -= w2 * c2[idx].x;
            r0.y -= w2 * c2[idx].y;
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "d2s",
)


ds_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ ds(float2* res, float2* c, float2* c1, float* r, float* mag, float* Deltar,
                   int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x       = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y       = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix      = (int)floorf(x);
    const int   iy      = (int)floorf(y);
    const float dx      = x - ix;
    const float dy      = y - iy;
    const float Deltarx = Deltar[2 * tz + 1];
    const float Deltary = Deltar[2 * tz + 0];
    const int   tz_off  = tz * npsi * nzpsi;

    // Precompute x-direction phi and dphi values (8 evals instead of 32).
    float px[4], dpx[4];
    for (int jx = -1; jx < 3; jx++) {
        float d = dx - jx;
        px[jx + 1]  = phi(d);
        dpx[jx + 1] = dphi(d);
    }

    float2 r0 = {};

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w   = dpx[jx + 1] * pdym  * Deltarx
                      + dpdym        * px[jx + 1] * Deltary;
            float w1  = px[jx + 1] * pdym;

            int   idx = indx_s + row_off;
            r0.x -= w * c[idx].x;
            r0.y -= w * c[idx].y;
            r0.x += w1 * c1[idx].x;
            r0.y += w1 * c1[idx].y;
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "ds",
)


dsadj_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dsadj(float2* f, float2* dt1, float2* dt2, float2* c, float2 *g, float* r, float* mag,
                      int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x      = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y      = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   tz_off = tz * npsi * nzpsi;
    const int   g_ind  = tx + ty * n + tz * n * nz;

    // Precompute x-direction phi and dphi values (8 evals instead of 32).
    float px[4], dpx[4];
    for (int jx = -1; jx < 3; jx++) {
        float d = dx - jx;
        px[jx + 1]  = phi(d);
        dpx[jx + 1] = dphi(d);
    }

    float2 g0 = g[g_ind];
    float2 dt10 = {};
    float2 dt20 = {};

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w1  = -dpdym       * px[jx + 1];
            float w2  = -dpx[jx + 1] * pdym;
            int   idx = indx_s + row_off;

            dt10.x += w1 * c[idx].x;
            dt10.y += w1 * c[idx].y;
            dt20.x += w2 * c[idx].x;
            dt20.y += w2 * c[idx].y;

            float w3 = px[jx + 1] * pdym;
            atomicAdd(&(f[idx].x), w3 * g0.x);
            atomicAdd(&(f[idx].y), w3 * g0.y);
        }
    }

    int out_ind = tx + ty * n + tz * n * nz;
    dt1[out_ind] = dt10;
    dt2[out_ind] = dt20;
}
}
""",
    "dsadj",
)


# ---------------------------------------------------------------------------
# Magnification-differentiating variants of ds / d2s / dsadj.
#
# The shift+demagnify operator samples psi at
#     x = mag_x * (tx - (n-1)/2) - r_x + (npsi-1)/2,
# so d/dmag_x = tau_x * d/dx and d/dr_x = -d/dx, i.e.
#     d/dmag_axis = -tau_axis(pixel) * d/dr_axis.
# A magnification perturbation Delta_m can therefore be folded into a per-pixel
# effective position perturbation Delta_r_eff = Delta_r - tau * Delta_m, which
# makes the m-derivative cost exactly one extra kernel launch instead of a
# separate sweep.  tau is the pixel coordinate measured from the tile centre.
# ---------------------------------------------------------------------------

dsm_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dsm(float2* res, float2* c, float2* c1,
                    float* r, float* mag, float* Deltar, float* Deltam,
                    int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x       = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y       = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix      = (int)floorf(x);
    const int   iy      = (int)floorf(y);
    const float dx      = x - ix;
    const float dy      = y - iy;

    // Effective per-pixel r-direction: Delta_r - tau * Delta_m.
    const float taux    = tx - (n  - 1) * 0.5f;
    const float tauy    = ty - (nz - 1) * 0.5f;
    const float Deltarx = Deltar[2 * tz + 1] - taux * Deltam[2 * tz + 1];
    const float Deltary = Deltar[2 * tz + 0] - tauy * Deltam[2 * tz + 0];
    const int   tz_off  = tz * npsi * nzpsi;

    float px[4], dpx[4];
    for (int jx = -1; jx < 3; jx++) {
        float d = dx - jx;
        px[jx + 1]  = phi(d);
        dpx[jx + 1] = dphi(d);
    }

    float2 r0 = {};

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w   = dpx[jx + 1] * pdym  * Deltarx
                      + dpdym        * px[jx + 1] * Deltary;
            float w1  = px[jx + 1] * pdym;

            int   idx = indx_s + row_off;
            r0.x -= w * c[idx].x;
            r0.y -= w * c[idx].y;
            r0.x += w1 * c1[idx].x;
            r0.y += w1 * c1[idx].y;
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "dsm",
)


d2sm_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + fun_d2phi
    + r"""
// SLOT PAIRING: w1 is built from Deltar1 and multiplies c1, w2 from Deltar2
// and multiplies c2 -- each coefficient is contracted with the shift in its
// OWN slot. The mixed second differential needs the opposite, so the CALLER
// passes the coefficients crossed (see Shift.d2curlySc and Rec.d2F_dF3). The
// unfused reference splits this into dT(c1, dr2) + dT(c2, dr1) + d2T(c), where
// the crossing is visible in the Python instead.
void __global__ d2sm(float2* res, float2* c, float2* c1, float2* c2, float* r, float* mag,
                     float* Deltar1, float* Deltam1, float* Deltar2, float* Deltam2,
                     int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x        = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y        = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix       = (int)floorf(x);
    const int   iy       = (int)floorf(y);
    const float dx       = x - ix;
    const float dy       = y - iy;

    const float taux     = tx - (n  - 1) * 0.5f;
    const float tauy     = ty - (nz - 1) * 0.5f;
    const float Deltar1x = Deltar1[2 * tz + 1] - taux * Deltam1[2 * tz + 1];
    const float Deltar1y = Deltar1[2 * tz + 0] - tauy * Deltam1[2 * tz + 0];
    const float Deltar2x = Deltar2[2 * tz + 1] - taux * Deltam2[2 * tz + 1];
    const float Deltar2y = Deltar2[2 * tz + 0] - tauy * Deltam2[2 * tz + 0];
    const float cross    = Deltar1x * Deltar2y + Deltar1y * Deltar2x;
    const int   tz_off   = tz * npsi * nzpsi;

    float px[4], dpx[4], d2px[4];
    for (int jx = -1; jx < 3; jx++) {
        float d   = dx - jx;
        px[jx + 1]   = phi(d);
        dpx[jx + 1]  = dphi(d);
        d2px[jx + 1] = d2phi(d);
    }

    float2 r0 = {};

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        float d2pdym  = d2phi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w  = d2px[jx + 1] * pdym    * Deltar1x * Deltar2x
                     + dpx[jx + 1]  * dpdym   * cross
                     + px[jx + 1]   * d2pdym  * Deltar1y * Deltar2y;
            float w1 = dpx[jx + 1] * pdym  * Deltar1x
                     + dpdym        * px[jx + 1] * Deltar1y;
            float w2 = dpx[jx + 1] * pdym  * Deltar2x
                     + dpdym        * px[jx + 1] * Deltar2y;
            int idx = indx_s + row_off;
            r0.x += w  * c[idx].x;
            r0.y += w  * c[idx].y;
            r0.x -= w1 * c1[idx].x;
            r0.y -= w1 * c1[idx].y;
            r0.x -= w2 * c2[idx].x;
            r0.y -= w2 * c2[idx].y;
        }
    }

    res[tx + ty * n + tz * n * nz] = r0;
}
}
""",
    "d2sm",
)


dsmadj_kernel = cp.RawKernel(
    r"""
extern "C"
{
"""
    + fun_phi
    + fun_dphi
    + r"""
void __global__ dsmadj(float2* f,
                       float2* dt1,  float2* dt2,
                       float2* dtm1, float2* dtm2,
                       float2* c, float2 *g, float* r, float* mag,
                       int n, int npsi, int nz, int nzpsi, int ntheta)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= n || ty >= nz || tz >= ntheta) return;

    const float x      = mag[2 * tz + 1] * (tx - (n-1) * 0.5f) - r[2 * tz + 1] + (npsi-1) * 0.5f;
    const float y      = mag[2 * tz + 0] * (ty - (nz-1) * 0.5f) - r[2 * tz + 0] + (nzpsi-1) * 0.5f;
    const int   ix     = (int)floorf(x);
    const int   iy     = (int)floorf(y);
    const float dx     = x - ix;
    const float dy     = y - iy;
    const int   tz_off = tz * npsi * nzpsi;
    const int   g_ind  = tx + ty * n + tz * n * nz;

    float px[4], dpx[4];
    for (int jx = -1; jx < 3; jx++) {
        float d = dx - jx;
        px[jx + 1]  = phi(d);
        dpx[jx + 1] = dphi(d);
    }

    float2 g0 = g[g_ind];
    float2 dt10 = {};
    float2 dt20 = {};

    for (int jy = -1; jy < 3; jy++)
    {
        int indy = iy + jy;
        int indy_s = sym_idx(indy, nzpsi);
        float dym     = dy - jy;
        float pdym    = phi(dym);
        float dpdym   = dphi(dym);
        int   row_off = indy_s * npsi + tz_off;

        for (int jx = -1; jx < 3; jx++)
        {
            int indx = ix + jx;
            int indx_s = sym_idx(indx, npsi);

            float w1  = -dpdym       * px[jx + 1];
            float w2  = -dpx[jx + 1] * pdym;
            int   idx = indx_s + row_off;

            dt10.x += w1 * c[idx].x;
            dt10.y += w1 * c[idx].y;
            dt20.x += w2 * c[idx].x;
            dt20.y += w2 * c[idx].y;

            float w3 = px[jx + 1] * pdym;
            atomicAdd(&(f[idx].x), w3 * g0.x);
            atomicAdd(&(f[idx].y), w3 * g0.y);
        }
    }

    // d/dmag_axis = -tau_axis * d/dr_axis, evaluated per pixel before the
    // redot reduction over (y, x) that turns these into scalars.
    const float tauy = ty - (nz - 1) * 0.5f;
    const float taux = tx - (n  - 1) * 0.5f;
    int out_ind = tx + ty * n + tz * n * nz;
    dt1 [out_ind] = dt10;
    dt2 [out_ind] = dt20;
    dtm1[out_ind].x = -tauy * dt10.x;
    dtm1[out_ind].y = -tauy * dt10.y;
    dtm2[out_ind].x = -taux * dt20.x;
    dtm2[out_ind].y = -taux * dt20.y;
}
}
""",
    "dsmadj",
)


