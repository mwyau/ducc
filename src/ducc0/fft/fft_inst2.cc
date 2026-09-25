#include "ducc0/fft/fftnd_impl.h"

namespace ducc0{
namespace detail_fft {
#define T double
#include "ducc0/fft/fft_inst_inc.h"
#undef T

#ifndef DUCC0_USE_NANOBIND
#define T long double
#include "ducc0/fft/fft_inst_inc.h"
#undef T
#endif
#undef DUCC0_FFTINST_ATTR
}
}
