#include "ducc0/fft/fftnd_impl.h"

namespace ducc0{
namespace detail_fft {
template Tcpass<float> cfftpass<float>::make_pass(size_t, size_t, size_t,
  const Troots<float> &, bool);
template Trpass<float> rfftpass<float>::make_pass(size_t, size_t, size_t,
  const Troots<float> &, bool);
#define T float
#include "ducc0/fft/fft_inst_inc.h"
#undef T
}
}
