#include <cfloat>
#include "ducc0/fft/fftnd_impl.h"

// Avoid duplicate double instantiations where long double aliases double.
#if LDBL_MANT_DIG > DBL_MANT_DIG
namespace ducc0 {
namespace detail_fft {

#if (defined(__GNUC__) || defined(__clang__)) && defined(__ELF__)
#define DUCC0_FFTINST_ATTR __attribute__((cold, section(".text.ducc0_cold")))
#elif defined(__GNUC__) || defined(__clang__)
#define DUCC0_FFTINST_ATTR __attribute__((cold))
#else
#define DUCC0_FFTINST_ATTR
#endif
#define T long double
#include "ducc0/fft/fft_inst_inc.h"
#undef T
#undef DUCC0_FFTINST_ATTR

}
}
#endif
