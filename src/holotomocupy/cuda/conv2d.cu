/*
 * conv2d_example.cu — 2D cuFFTDx convolution (single size, JIT-compiled).
 *
 * NX, NY injected via -DNX=... -DNY=... at compile time by Python.
 * EPT/FPB are auto-computed from NX, NY at compile time.
 *
 * Pipeline: FFT_Y -> (FFT_X -> H*x -> IFFT_X) -> IFFT_Y
 *
 * API:
 *   void* conv2d_create(cudaStream_t stream)
 *   void  conv2d_run(void* h, const void* x, const void* H, void* y, int batches, cudaStream_t)
 *   void  conv2d_destroy(void* h)
 *
 * Arrays: x, y are [batches, NX, NY] complex64;  H is [NX, NY] complex64.
 */

#include <optional>
#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "common/common.hpp"
#include "07_convolution_3d/kernels.hpp"
#include "io_2d_conv_smem.hpp"

using namespace cufftdx;
using namespace example;

// ---------------------------------------------------------------------------
// Precision
// ---------------------------------------------------------------------------
static constexpr unsigned SM_VER = 800;
using fp_t      = float;
using complex_t = cufftdx::complex<fp_t>;

// NX, NY, EPT_X, FPB_X, EPT_Y, FPB_Y are all injected via -D flags.

// ---------------------------------------------------------------------------
// FFT descriptors
// ---------------------------------------------------------------------------
using FFTXPartial = decltype(Block() + Size<NX>() + Type<fft_type::c2c>() +
                              ElementsPerThread<EPT_X>() + FFTsPerBlock<FPB_X>() +
                              Precision<fp_t>() + SM<SM_VER>());
using FFTYPartial = decltype(Block() + Size<NY>() + Type<fft_type::c2c>() +
                              ElementsPerThread<EPT_Y>() + FFTsPerBlock<FPB_Y>() +
                              Precision<fp_t>() + SM<SM_VER>());

using FFTX_FRONT = decltype(FFTXPartial() + Direction<fft_direction::forward>());
using FFTX_BACK  = decltype(FFTXPartial() + Direction<fft_direction::inverse>());
using FFTY_FRONT = decltype(FFTYPartial() + Direction<fft_direction::forward>());
using FFTY_BACK  = decltype(FFTYPartial() + Direction<fft_direction::inverse>());

// io_2d_conv_smem is defined in io_2d_conv_smem.hpp
using example::io_2d_conv_smem;

using IO_X = io_2d_conv_smem<dimension::x, true, 1, FFTX_FRONT, FFTX_BACK, FFTY_FRONT, FFTY_BACK>;
using IO_Y = io_2d_conv_smem<dimension::y, true, 1, FFTX_FRONT, FFTX_BACK, FFTY_FRONT, FFTY_BACK>;

static constexpr int      x_smem = IO_X::get_shared_bytes();
static constexpr int      y_smem = IO_Y::get_shared_bytes();
static constexpr unsigned x_sub  = NY;   // NY X-FFTs per batch
static constexpr unsigned y_sub  = NX;   // NX Y-FFTs per batch

// ---------------------------------------------------------------------------
// X convolution kernel: FFT_X -> H[ix*NY + iy] multiply -> IFFT_X
// ---------------------------------------------------------------------------
template<class FFT, class IFFT, class IOF, class IOB, unsigned SY>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void conv_kernel_H(int sub, typename FFT::input_type* in,
                               typename IFFT::output_type* out,
                               typename FFT::workspace_type ws,
                               const typename FFT::value_type* H)
{
    using ct = typename FFT::value_type;
    if (threadIdx.y + blockIdx.x * FFT::ffts_per_block >= sub) return;
    ct td[FFT::storage_size];
    extern __shared__ __align__(16) ct smem[];

    IOF{}.load_gmem_to_rmem(in, smem, td);
    __syncthreads();
    FFT().execute(td, smem, ws);
    __syncthreads();

    const unsigned iy = threadIdx.y + blockIdx.x * FFT::ffts_per_block;
    using vt = typename FFT::output_type;
#pragma unroll
    for (int i = 0; i < FFT::output_ept; ++i) {
        unsigned ix = threadIdx.x + i * FFT::stride;
        auto& v = reinterpret_cast<vt*>(td)[i];
        auto  h = H[ix * SY + iy];
        v = vt{v.x*h.x - v.y*h.y, v.x*h.y + v.y*h.x};
    }

    IFFT().execute(td, smem, ws);
    __syncthreads();
    IOB{}.store_rmem_to_gmem(td, smem, out);
}

// ---------------------------------------------------------------------------
// Handle and API
// ---------------------------------------------------------------------------

// Workspace lives as static auto locals in conv2d_run (see below).
struct Conv2DHandle {};

// Returns actual threads-per-FFT (block_dim.x) so Python can validate and recompile if needed.
extern "C" void conv2d_block_dim(int* bx, int* by) {
    *bx = (int)FFTY_FRONT::block_dim.x;
    *by = (int)FFTY_FRONT::block_dim.y;
}

// Returns the dynamic shared memory required by each kernel pair.
// Used to detect cases where smem > device max (163840 on sm_80) and reduce FPB.
extern "C" void conv2d_smem_size(int* xs, int* ys) {
    *xs = (int)x_smem;
    *ys = (int)y_smem;
}

extern "C" void* conv2d_create(cudaStream_t stream) {
    static bool done = false;
    if (!done) {
        done = true;
        cudaFuncSetAttribute(fft_kernel<FFTY_FRONT, IO_Y, identity, identity, complex_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, y_smem);
        cudaFuncSetAttribute(conv_kernel_H<FFTX_FRONT, FFTX_BACK, IO_X, IO_X, NY>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, x_smem);
        cudaFuncSetAttribute(fft_kernel<FFTY_BACK, IO_Y, identity, identity, complex_t>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, y_smem);
    }
    return new Conv2DHandle;
}

extern "C" void conv2d_run(void* handle, const void* x_ptr, const void* H_ptr,
                            void* y_ptr, int batches, cudaStream_t stream) {
    // Static auto: avoids naming the private host_workspace_type.
    // Initialized once on first call (with the caller's stream); device memory
    // persists for the .so lifetime, so the device_handle stays valid.
    static auto ws_x = [&]{ cudaError_t e; return cufftdx::make_workspace<FFTX_FRONT>(e, stream); }();
    static auto ws_y = [&]{ cudaError_t e; return cufftdx::make_workspace<FFTY_FRONT>(e, stream); }();

    auto* x = static_cast<const complex_t*>(x_ptr);
    auto* H = static_cast<const complex_t*>(H_ptr);
    auto* y = static_cast<complex_t*>(y_ptr);
    const unsigned ub = (unsigned)batches;

    fft_kernel<FFTY_FRONT, IO_Y, identity, identity, complex_t>
        <<<dim3{div_up(y_sub, (unsigned)FPB_Y), ub, 1}, FFTY_FRONT::block_dim, y_smem, stream>>>(
        y_sub, const_cast<complex_t*>(x), y, ws_y);

    conv_kernel_H<FFTX_FRONT, FFTX_BACK, IO_X, IO_X, NY>
        <<<dim3{div_up(x_sub, (unsigned)FPB_X), ub, 1}, FFTX_FRONT::block_dim, x_smem, stream>>>(
        x_sub, y, y, ws_x, H);

    fft_kernel<FFTY_BACK, IO_Y, identity, identity, complex_t>
        <<<dim3{div_up(y_sub, (unsigned)FPB_Y), ub, 1}, FFTY_BACK::block_dim, y_smem, stream>>>(
        y_sub, y, y, ws_y);
}

extern "C" void conv2d_destroy(void* handle) {
    delete static_cast<Conv2DHandle*>(handle);
}
