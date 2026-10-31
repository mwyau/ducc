// Experimental, scalar-execution, intra-transform AVX-512 butterflies.
// The surrounding DUCC 1D/ND SIMD-across-transforms dispatch is unchanged.
// Only include inside detail_fft's anonymous namespace after Cmplx is defined.
#pragma once
#if defined(__AVX512F__) && !defined(DUCC0_NO_SIMD)
template<typename T> struct avx512_butterfly_ops;

template<> struct avx512_butterfly_ops<float>
  {
  using V = __m512;
  static constexpr size_t ncomplex=8;
  static V load(const Cmplx<float> *p)
    { return _mm512_loadu_ps(reinterpret_cast<const float *>(p)); }
  static void store(Cmplx<float> *p, V x)
    { _mm512_storeu_ps(reinterpret_cast<float *>(p),x); }
  static V add(V a,V b) { return _mm512_add_ps(a,b); }
  static V sub(V a,V b) { return _mm512_sub_ps(a,b); }
  static V constant(float r,float i)
    { return _mm512_setr_ps(r,i,r,i,r,i,r,i,r,i,r,i,r,i,r,i); }
  template<bool fwd> static V rot90(V x)
    {
    auto y=_mm512_permute_ps(x,0xB1);
    const auto sign= fwd
      ? _mm512_setr_ps(0.f,-0.f,0.f,-0.f,0.f,-0.f,0.f,-0.f,
                       0.f,-0.f,0.f,-0.f,0.f,-0.f,0.f,-0.f)
      : _mm512_setr_ps(-0.f,0.f,-0.f,0.f,-0.f,0.f,-0.f,0.f,
                       -0.f,0.f,-0.f,0.f,-0.f,0.f,-0.f,0.f);
    return _mm512_castsi512_ps(_mm512_xor_si512(
      _mm512_castps_si512(y),_mm512_castps_si512(sign)));
    }
  static V mul(V a,V w)
    {
    auto wr=_mm512_moveldup_ps(w);
    auto wi=_mm512_movehdup_ps(w);
    return _mm512_fmaddsub_ps(a,wr,
      _mm512_mul_ps(_mm512_permute_ps(a,0xB1),wi));
    }
  template<size_t R,bool fwd> static V twiddle(
    const Cmplx<float> *wa,size_t i,size_t j)
    {
    auto w=[&](size_t lane)->Cmplx<float>
      { return wa[(i+lane-1)*(R-1)+(j-1)]; };
    const auto a=w(0),b=w(1),c=w(2),d=w(3);
    const auto e=w(4),f=w(5),g=w(6),h=w(7);
    const float sign=fwd?-1.f:1.f;
    return _mm512_setr_ps(a.r,sign*a.i,b.r,sign*b.i,
                          c.r,sign*c.i,d.r,sign*d.i,
                          e.r,sign*e.i,f.r,sign*f.i,
                          g.r,sign*g.i,h.r,sign*h.i);
    }
  };

template<> struct avx512_butterfly_ops<double>
  {
  using V = __m512d;
  static constexpr size_t ncomplex=4;
  static V load(const Cmplx<double> *p)
    { return _mm512_loadu_pd(reinterpret_cast<const double *>(p)); }
  static void store(Cmplx<double> *p,V x)
    { _mm512_storeu_pd(reinterpret_cast<double *>(p),x); }
  static V add(V a,V b) { return _mm512_add_pd(a,b); }
  static V sub(V a,V b) { return _mm512_sub_pd(a,b); }
  static V constant(double r,double i)
    { return _mm512_setr_pd(r,i,r,i,r,i,r,i); }
  template<bool fwd> static V rot90(V x)
    {
    auto y=_mm512_permute_pd(x,0x55);
    const auto sign=fwd
      ? _mm512_setr_pd(0.,-0.,0.,-0.,0.,-0.,0.,-0.)
      : _mm512_setr_pd(-0.,0.,-0.,0.,-0.,0.,-0.,0.);
    return _mm512_castsi512_pd(_mm512_xor_si512(
      _mm512_castpd_si512(y),_mm512_castpd_si512(sign)));
    }
  static V mul(V a,V w)
    {
    auto wr=_mm512_movedup_pd(w);
    auto wi=_mm512_permute_pd(w,0xFF);
    return _mm512_fmaddsub_pd(a,wr,
      _mm512_mul_pd(_mm512_permute_pd(a,0x55),wi));
    }
  template<size_t R,bool fwd> static V twiddle(
    const Cmplx<double> *wa,size_t i,size_t j)
    {
    auto w=[&](size_t lane)->Cmplx<double>
      { return wa[(i+lane-1)*(R-1)+(j-1)]; };
    const auto a=w(0),b=w(1),c=w(2),d=w(3);
    const double sign=fwd?-1.:1.;
    return _mm512_setr_pd(a.r,sign*a.i,b.r,sign*b.i,
                          c.r,sign*c.i,d.r,sign*d.i);
    }
  };

template<typename O,bool fwd> inline void avx512_dft4(
  typename O::V a,typename O::V b,typename O::V c,typename O::V d,
  typename O::V &y0,typename O::V &y1,
  typename O::V &y2,typename O::V &y3)
  {
  const auto t0=O::add(a,c), t2=O::sub(a,c);
  const auto t1=O::add(b,d), t3=O::template rot90<fwd>(O::sub(b,d));
  y0=O::add(t0,t1); y2=O::sub(t0,t1);
  y1=O::add(t2,t3); y3=O::sub(t2,t3);
  }

// Only for the scalar Cmplx<T> execution of cfftp4/cfftp8 with
// l1>1, ido>1.  i=0 and the residual columns remain in DUCC's kernel.
// Return the first unprocessed column; never process an incomplete vector.
template<size_t R,typename T,bool fwd>
inline size_t avx512_butterfly_columns(const Cmplx<T> *cc,
  Cmplx<T> *ch,const Cmplx<T> *wa,size_t ido,size_t l1,
  size_t k,size_t first)
  {
  static_assert(R==4 || R==8);
  using O=avx512_butterfly_ops<T>;
  using V=typename O::V;
  const size_t width=O::ncomplex;
  size_t i=first;
  for (;i+width<=ido;i+=width)
    {
    V x[R],y[R];
    for (size_t j=0;j<R;++j)
      x[j]=O::load(cc+i+ido*(j+R*k));
    if constexpr (R==4)
      avx512_dft4<O,fwd>(x[0],x[1],x[2],x[3],y[0],y[1],y[2],y[3]);
    else
      {
      V even[4],odd[4];
      avx512_dft4<O,fwd>(x[0],x[2],x[4],x[6],
        even[0],even[1],even[2],even[3]);
      avx512_dft4<O,fwd>(x[1],x[3],x[5],x[7],
        odd[0],odd[1],odd[2],odd[3]);
      constexpr T h=T(0.707106781186547524400844362104849L);
      odd[1]=O::mul(odd[1],O::constant(h,fwd?-h:h));
      odd[2]=O::template rot90<fwd>(odd[2]);
      odd[3]=O::mul(odd[3],O::constant(-h,fwd?-h:h));
      for(size_t j=0;j<4;++j)
        {
        y[j]=O::add(even[j],odd[j]);
        y[j+4]=O::sub(even[j],odd[j]);
        }
      }
    O::store(ch+i+ido*k,y[0]);
    for(size_t j=1;j<R;++j)
      O::store(ch+i+ido*(k+l1*j),
        O::mul(y[j],O::template twiddle<R,fwd>(wa,i,j)));
    }
  return i;
  }
#endif
