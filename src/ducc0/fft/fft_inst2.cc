#include <cfloat>
#include "ducc0/fft/fftnd_impl.h"

namespace ducc0{
namespace detail_fft {
#define T double
#include "ducc0/fft/fft_inst_inc.h"
#undef T

#if !defined(DUCC0_USE_NANOBIND) || LDBL_MANT_DIG > DBL_MANT_DIG
#define T long double
#include "ducc0/fft/fft_inst_inc.h"
#undef T
#endif
}
}
