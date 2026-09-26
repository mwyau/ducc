/* Compile-time FFT backend namespace and table symbol for one ISA profile. */
#ifndef DUCC0_FFT_TARGET_PROFILE_H
#define DUCC0_FFT_TARGET_PROFILE_H

#ifndef DUCC0_DISPATCH_TARGET
#  define DUCC0_DISPATCH_TARGET 1
#endif

#if defined(DUCC0_X86_64_LEVEL)
#  define DUCC0_FFT_PROFILE_LEVEL DUCC0_X86_64_LEVEL
#elif defined(__AVX512F__)
#  define DUCC0_FFT_PROFILE_LEVEL 4
#elif defined(__AVX2__)
#  define DUCC0_FFT_PROFILE_LEVEL 3
#elif defined(__SSE4_2__)
#  define DUCC0_FFT_PROFILE_LEVEL 2
#else
#  define DUCC0_FFT_PROFILE_LEVEL 1
#endif

#ifndef DUCC0_X86_64_LEVEL
#  define DUCC0_X86_64_LEVEL DUCC0_FFT_PROFILE_LEVEL
#endif

#if DUCC0_FFT_PROFILE_LEVEL == 1
#  define DUCC0_FFT_NAMESPACE detail_fft_target_x86_64
#  define DUCC0_FFT_KERNELS_F32 fft_kernels_x86_64_f32
#  define DUCC0_FFT_KERNELS_F64 fft_kernels_x86_64_f64
#elif DUCC0_FFT_PROFILE_LEVEL == 2
#  define DUCC0_FFT_NAMESPACE detail_fft_target_x86_64_v2
#  define DUCC0_FFT_KERNELS_F32 fft_kernels_x86_64_v2_f32
#  define DUCC0_FFT_KERNELS_F64 fft_kernels_x86_64_v2_f64
#elif DUCC0_FFT_PROFILE_LEVEL == 3
#  define DUCC0_FFT_NAMESPACE detail_fft_target_x86_64_v3
#  define DUCC0_FFT_KERNELS_F32 fft_kernels_x86_64_v3_f32
#  define DUCC0_FFT_KERNELS_F64 fft_kernels_x86_64_v3_f64
#elif DUCC0_FFT_PROFILE_LEVEL == 4
#  define DUCC0_FFT_NAMESPACE detail_fft_target_x86_64_v4
#  define DUCC0_FFT_KERNELS_F32 fft_kernels_x86_64_v4_f32
#  define DUCC0_FFT_KERNELS_F64 fft_kernels_x86_64_v4_f64
#else
#  error "Invalid DUCC0 FFT profile level"
#endif

#endif
