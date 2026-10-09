#pragma once
// Shared memory IO helper for 2D FFT/convolution kernels.
// Extracted from conv2d_example.cu so both conv2d and fft2d can use it.

#include <type_traits>
#include <algorithm>
#include <cufftdx.hpp>
#include "07_convolution_3d/index_mapper.hpp"
#include "common/common.hpp"

namespace example {

template<dimension Dim, bool Front, int Batches,
         class FFTX_, class IFFTX_, class FFTY_, class IFFTY_>
class io_2d_conv_smem {
    using FFTX = std::conditional_t<Front, FFTX_, IFFTX_>;
    using FFTY = std::conditional_t<Front, FFTY_, IFFTY_>;
    using vt   = typename FFTX::value_type;
    static_assert(std::is_same_v<vt, typename FFTY::value_type>);

    static constexpr unsigned sx = cufftdx::size_of<FFTX>::value;
    static constexpr unsigned sy = cufftdx::size_of<FFTY>::value;
    static constexpr unsigned flat = sx * sy;

    static constexpr auto xfpb = FFTX::ffts_per_block;
    static constexpr auto xpad = (example::warp_size + xfpb - 1) / xfpb;

    using gl_x = index_mapper<int_pair<sx, sy>, int_pair<sy, 1>, int_pair<Batches, flat>>;
    using sl_x = index_mapper<int_pair<sx, 1>,  int_pair<xfpb, sx + xpad>>;

    static constexpr int x_smem = std::max<int>(FFTX::shared_memory_size,
        (int)(sx * xfpb + xfpb * xpad) * (int)sizeof(vt));

    template<class FFT, int Sub, class GL, class SL>
    __device__ __forceinline__ void gmem2smem(const vt* g, vt* s) const {
        GL gl; SL sl;
        constexpr auto fpb = FFT::ffts_per_block;
        const auto bfpb = (blockIdx.x == Sub / fpb) ? Sub % fpb : fpb;
        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
        const int re = tid / bfpb, rb = tid % bfpb;
        auto ig = reinterpret_cast<const typename FFT::input_type*>(g);
        auto is = reinterpret_cast<      typename FFT::input_type*>(s);
#pragma unroll
        for (int i = 0; i < FFT::input_ept; ++i) {
            int e = re + i * FFT::stride, gib = rb + blockIdx.x * fpb;
            if (not FFT::requires_workspace or e < (int)FFT::input_length)
                is[sl(e, rb)] = ig[gl(e, gib, blockIdx.y)];
        }
    }
    template<class FFT, int Sub, class SL, class GL>
    __device__ __forceinline__ void smem2gmem(const vt* s, vt* g) const {
        GL gl; SL sl;
        constexpr auto fpb = FFT::ffts_per_block;
        const auto bfpb = (blockIdx.x == Sub / fpb) ? Sub % fpb : fpb;
        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
        const int re = tid / bfpb, rb = tid % bfpb;
        auto og = reinterpret_cast<      typename FFT::output_type*>(g);
        auto os = reinterpret_cast<const typename FFT::output_type*>(s);
#pragma unroll
        for (int i = 0; i < FFT::output_ept; ++i) {
            int e = re + i * FFT::stride, gib = rb + blockIdx.x * fpb;
            if (not FFT::requires_workspace or e < (int)FFT::output_length)
                og[gl(e, gib, blockIdx.y)] = os[sl(e, rb)];
        }
    }
    template<class FFT, class SL, class Op>
    __device__ __forceinline__ void smem2rmem(const vt* s, vt* r) const {
        SL sl; Op op;
        auto ir = reinterpret_cast<      typename FFT::input_type*>(r);
        auto is = reinterpret_cast<const typename FFT::input_type*>(s);
#pragma unroll
        for (int i = 0; i < FFT::input_ept; ++i) {
            int e = threadIdx.x + i * FFT::stride;
            if (not FFT::requires_workspace or e < (int)FFT::input_length)
                ir[i] = op(is[sl(e, threadIdx.y)]);
        }
    }
    template<class FFT, class SL, class Op>
    __device__ __forceinline__ void rmem2smem(const vt* r, vt* s) const {
        SL sl; Op op;
        auto os   = reinterpret_cast<      typename FFT::output_type*>(s);
        auto outr = reinterpret_cast<const typename FFT::output_type*>(r);
#pragma unroll
        for (int i = 0; i < FFT::output_ept; ++i) {
            int e = threadIdx.x + i * FFT::stride;
            if (not FFT::requires_workspace or e < (int)FFT::output_length)
                os[sl(e, threadIdx.y)] = op(outr[i]);
        }
    }

public:
    static constexpr __host__ __device__ size_t get_shared_bytes() {
        return (Dim == dimension::x) ? (size_t)x_smem : FFTY::shared_memory_size;
    }

    template<typename GT, typename ST, typename RT, class Op = identity>
    __device__ __forceinline__
    void load_gmem_to_rmem(const GT* g, [[maybe_unused]] ST* s, RT* r,
                            [[maybe_unused]] Op op = {}) const {
        if constexpr (Dim == dimension::x) {
            gmem2smem<FFTX, (int)sy, gl_x, sl_x>(
                reinterpret_cast<const vt*>(g), reinterpret_cast<vt*>(s));
            __syncthreads();
            smem2rmem<FFTX, sl_x, Op>(reinterpret_cast<const vt*>(s), reinterpret_cast<vt*>(r));
        } else {
            example::io<FFTY>::load(
                reinterpret_cast<const typename FFTY::input_type*>(g) + blockIdx.y * flat,
                reinterpret_cast<vt*>(r), threadIdx.y, op);
        }
    }
    template<typename RT, typename ST, typename GT, class Op = identity>
    __device__ __forceinline__
    void store_rmem_to_gmem(const RT* r, [[maybe_unused]] ST* s, GT* g,
                             [[maybe_unused]] Op op = {}) const {
        if constexpr (Dim == dimension::x) {
            rmem2smem<FFTX, sl_x, Op>(reinterpret_cast<const vt*>(r), reinterpret_cast<vt*>(s));
            __syncthreads();
            smem2gmem<FFTX, (int)sy, sl_x, gl_x>(reinterpret_cast<const vt*>(s), reinterpret_cast<vt*>(g));
        } else {
            example::io<FFTY>::store(
                reinterpret_cast<const vt*>(r),
                reinterpret_cast<typename FFTY::output_type*>(g) + blockIdx.y * flat,
                threadIdx.y, op);
        }
    }
};

} // namespace example
