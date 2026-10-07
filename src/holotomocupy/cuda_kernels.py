import cupy as cp

gather_kernel = cp.RawKernel(
    r"""
extern "C" __global__ void gather(float2* g, float2* f, float* theta, int m, float* mu,
                                  int n, int ndet, int ntheta, int nz, bool dir)
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

    // Samples whose Cartesian coordinate falls OUTSIDE the padded FFT square are
    // skipped, not wrapped.  Such samples only exist once ndet > n.
    //
    // Why the test is on (x0, y0) and not on |fr|.  The object lives on an n x n
    // grid, so the spectrum it can represent is a SQUARE, |kx| < 1/2 and
    // |ky| < 1/2 -- not a disc.  A radial line at angle theta leaves that square
    // at |fr| = 1/(2*max(|cos|,|sin|)): 1/2 along the axes, but sqrt(2)/2 = 0.707
    // at 45 deg.  Guarding on |fr| >= 1/2 would therefore discard the corners of
    // the square at every oblique angle -- real, recoverable content.  Guarding
    // on the Cartesian coordinates discards exactly the samples that have no
    // array cell to read, and keeps the corners.
    //
    // What this replaces.  The index used to wrap ((n + ell + 2n) % 2n), which
    // models the object as a delta comb on the n grid, so out-of-square bins
    // carried aliased replicas of the low frequencies.  That is what
    // ~/APS_PXM/tomo_usfft does, where it is invisible because that package
    // generates its data with this same R (an inverse crime).
    //
    // The wrap was not a mild error.  At theta = 0 and 90 -- and only there --
    // it is exact rather than aliasing: the shift between bins tx and tx+n is
    // 2n*cos(theta) cells, which is 0 (mod 2n) only for cos/sin in {0,+-1}, so
    // those two rows read back the identical spectrum, are n-periodic, and come
    // out of the length-ndet inverse transform as combs with every odd detector
    // sample exactly zero.  Measured at ndet == 2n: |odd|/|even| is 0.00e+00 at
    // theta = 0 and 90 against 0.999 median over all other angles, and half of
    // those two rows' energy sits above the object's Nyquist.  Two projections
    // per scan are destroyed, and they are the two aligned with x and y -- which
    // is why the ctxl tomo_upsample=2 reconstructions carried high-frequency
    // vertical and horizontal line artifacts (300x the background power on the
    // two Fourier axes, against 1.0x for the tomo_upsample=1 arm) while every
    // other angle looked fine.  (An integer shift is not enough to do this:
    // theta = atan(3/4) at 2n = 320 shifts by a whole 256 cells and still
    // aliases.)
    //
    // ndet == n is unchanged.  There fr is in [-1/2, 1/2], so |x0|, |y0| <= 1/2
    // and the guard -- strict on the upper side, see below -- can never fire.
    // Regression test at n = 128: R max|diff| = 0 exactly, RT max|diff| =
    // 1.18e-08, which is the same figure the old kernel gets against ITSELF on
    // the same input (the scatter's atomicAdd is order-nondeterministic), i.e.
    // bit-for-bit to the extent RT is ever bit-for-bit.
    //
    // NOTE the % twon in the loops below stays.  That one is the interpolation
    // stencil straddling the array edge -- correct periodic evaluation of a
    // spectrum that really is periodic, exercised at ndet == n too.  Only the
    // sample CENTRE leaving the square is aliasing; the stencil skirt is not.

    const float x0 =  fr * __cosf(theta[ty]);
    const float y0 = -fr * __sinf(theta[ty]);

    // Strict > on the upper side: x0 == +1/2 and x0 == -1/2 are the SAME
    // frequency for a periodic spectrum, so the lone boundary bin (fr = -1/2,
    // which lands on +1/2 for theta > 90 deg) is legitimately wrapped, not
    // aliased.  Rejecting it would change ndet == n, where fr = -1/2 is the
    // tx = 0 bin of every projection.
    if (x0 < -0.5f || x0 > 0.5f || y0 < -0.5f || y0 > 0.5f)
    {
        if (dir == 0) g[g_ind] = make_float2(0.0f, 0.0f);
        return;
    }

    // ROTATION AXIS AT (N-1)/2, NOT N/2.
    // The fftshift-by-sign convention puts the object's rotation centre at
    // index n/2 and the detector origin at nd/2.  n/2 is not the geometric
    // middle of pixels 0..n-1, so that axis sits at physical N0/2 + 0.5/scale
    // -- it DRIFTS when the data is binned, and the reader had to cancel the
    // drift with a 0.5*(scale-1) term.  Referencing both to (N-1)/2 pins the
    // axis at the detector middle for every bin level, so a shift measured
    // from the middle just scales by 2.
    //
    // Current minus wanted is a per-angle detector shift
    //     d(theta) = (cos - sin)/2 - 1/2        (0 at theta=0, -1 at 90 deg)
    // and undoing it is one phase on the gathered sample.  dir==1 applies the
    // conjugate before scattering, so R and RT stay an exact adjoint pair.
    const float d_ax = 0.5f * (__cosf(theta[ty]) - __sinf(theta[ty])) - 0.5f;
    float sn_ax, cs_ax;
    __sincosf(-6.283185307179586f * fr * d_ax, &sn_ax, &cs_ax);

    float2 g0;
    if (dir == 0) {
        g0 = make_float2(0.0f, 0.0f);
    } else {
        const float2 gin = g[g_ind];                       // conj(phase) * gin
        g0 = make_float2( cs_ax * gin.x + sn_ax * gin.y,
                         -sn_ax * gin.x + cs_ax * gin.y);
    }

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

            if (dir == 0)
            {
                g0.x += w * f[f_ind].x;
                g0.y += w * f[f_ind].y;
            }
            else
            {
                atomicAdd(&(f[f_ind].x), w * g0.x);
                atomicAdd(&(f[f_ind].y), w * g0.y);
            }
        }
    }

    if (dir == 0)
    {
        const float gx = cs_ax * g0.x - sn_ax * g0.y;      // phase * g0
        const float gy = sn_ax * g0.x + cs_ax * g0.y;
        g[g_ind].x = gx / n;
        g[g_ind].y = gy / n;
    }
}
""",
    "gather",
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


