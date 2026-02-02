#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <memory>
#include "gint/sycl_alloc.hpp"
#include <exchcxx/xc_kernel.hpp>         // ExchCXX
#include "exchcxx.h"                     // ABI structs

/* ---------------- Version / reference ---------------- */
static const char* kRef   = "ExchCXX GPU shim (libxc ABI)";
static const char* kDOI   = "";
static const char* kKey   = "ExchCXX";
static const char* kVers  = "ExchCXX-SYCL 1.0";

extern "C" {
const char *xc_reference(void)      { return kRef; }
const char *xc_reference_doi(void)  { return kDOI; }
const char *xc_reference_key(void)  { return kKey; }
void xc_version(int *maj,int *min,int *mic){ if(maj) *maj=1; if(min)*min=0; if(mic)*mic=0; }
const char *xc_version_string(void) { return kVers; }
}

/* ---------------- Name -> ExchCXX::Functional map ---------------- */
/* Extend as needed; names are case-insensitive. */
static ExchCXX::Functional map_name_to_func(const std::string& s_in, int* family_out, bool* needs_lapl_out) {
  std::string s = s_in;
  for(auto& c : s) c = ::toupper(c);

  // LDA
  if(s == "SVWN5" || s == "LDA" || s == "LDA_XC_VWN_5") {
    if(family_out) *family_out = XC_FAMILY_LDA;
    if(needs_lapl_out) *needs_lapl_out = false;
    return ExchCXX::Functional::SVWN5;
  }
  // GGA
  if(s == "PBE" || s == "PBE_XC" || s == "PBE_XC_PBE") {
    if(family_out) *family_out = XC_FAMILY_GGA;
    if(needs_lapl_out) *needs_lapl_out = false;
    return ExchCXX::Functional::PBE;
  }
  if(s == "BLYP") {
    if(family_out) *family_out = XC_FAMILY_GGA;
    if(needs_lapl_out) *needs_lapl_out = false;
    return ExchCXX::Functional::BLYP;
  }
  if(s == "PBE0") {
    if(family_out) *family_out = XC_FAMILY_GGA; // hybrid GGA, but family used by Python logic
    if(needs_lapl_out) *needs_lapl_out = false;
    return ExchCXX::Functional::PBE0;
  }
  // MGGA (tau-only)
  if(s == "SCAN") {
    if(family_out) *family_out = XC_FAMILY_MGGA;
    if(needs_lapl_out) *needs_lapl_out = false;
    return ExchCXX::Functional::SCAN;
  }
  if(s == "M06-2X" || s == "M062X") {
    if(family_out) *family_out = XC_FAMILY_MGGA;
    if(needs_lapl_out) *needs_lapl_out = false;
    return ExchCXX::Functional::M062X;
  }
  // MGGA (needs laplacian)
  if(s == "R2SCANL") {
    if(family_out) *family_out = XC_FAMILY_MGGA;
    if(needs_lapl_out) *needs_lapl_out = true;
    return ExchCXX::Functional::R2SCANL;
  }

  // Fallback – treat as PBE
  if(family_out) *family_out = XC_FAMILY_GGA;
  if(needs_lapl_out) *needs_lapl_out = false;
  return ExchCXX::Functional::PBE;
}

extern "C" int xc_functional_get_number(const char *name) {
  // We don’t use numeric IDs internally; return a stable pseudo-id.
  if(!name) return 0;
  int family=0; bool need_lapl=false;
  auto f = map_name_to_func(name, &family, &need_lapl);
  // Simple hash: family in high, functional enum in low
  return (family << 16) | static_cast<int>(f);
}

/* ---------------- Shim state kept in xc_func_type::params ---------------- */
struct ShimImpl {
  ExchCXX::Spin spin;
  int family;            // XC_FAMILY_*
  bool needs_lapl;
  std::unique_ptr<ExchCXX::XCKernel> k;  // single kernel per xc_func_type
};

static inline ShimImpl* get_impl(const xc_func_type* p){
  return reinterpret_cast<ShimImpl*>(p ? p->params : nullptr);
}

/* Populate minimal dimensions for the arrays we actually use.
   Others left as 0 so Python won’t allocate them unless requested. */
static void fill_dimensions(xc_dimensions* d, int family, int nspin, bool needs_lapl) {
  memset(d, 0, sizeof(*d));
  const int rho_dim   = (nspin == XC_UNPOLARIZED) ? 1 : 2;
  const int sigma_dim = (nspin == XC_UNPOLARIZED) ? 1 : 3;  // (aa,ab,bb) when pol
  const int lapl_dim  = rho_dim;
  const int tau_dim   = rho_dim;

  d->rho = rho_dim;
  d->sigma = sigma_dim;
  d->lapl = lapl_dim;
  d->tau = tau_dim;

  d->zk = 1;
  d->vrho = rho_dim;

  if(family >= XC_FAMILY_GGA) {
    d->vsigma = sigma_dim;
  }
  if(family >= XC_FAMILY_MGGA) {
    d->vtau = tau_dim;
    d->vlapl = needs_lapl ? lapl_dim : 0;
  }

  /* Higher orders remain zero; we currently implement EXC/VXC only. */
}

/* ---------------- Minimal libxc-like lifecycle ---------------- */
extern "C" xc_func_type *xc_func_alloc(void) {
  auto *p = (xc_func_type*) std::calloc(1, sizeof(xc_func_type));
  return p;
}

extern "C" int xc_func_init(xc_func_type *p, int functional, int nspin) {
  if(!p) return 1;
  if(nspin != XC_UNPOLARIZED && nspin != XC_POLARIZED) return 2;

  // Reconstruct our functional choice from the pseudo-id if given, else default PBE
  ExchCXX::Functional f = ExchCXX::Functional::PBE;
  int family = XC_FAMILY_GGA;
  bool needs_lapl = false;

  // If functional came from xc_functional_get_number, it encodes family in high bits
  if(functional != 0) {
    family = (functional >> 16) & 0xFFFF;
    f = static_cast<ExchCXX::Functional>(functional & 0xFFFF);
    // Guess needs_lapl for known ones
    if(f == ExchCXX::Functional::R2SCANL) needs_lapl = true;
  }

  auto impl = std::make_unique<ShimImpl>();
  impl->spin = (nspin == XC_UNPOLARIZED) ? ExchCXX::Spin::Unpolarized : ExchCXX::Spin::Polarized;
  impl->family = family;
  impl->needs_lapl = needs_lapl;

  try {
    impl->k = std::make_unique<ExchCXX::XCKernel>(ExchCXX::Backend::builtin, f, impl->spin);
  } catch(const std::exception& e){
    std::fprintf(stderr, "ExchCXX kernel construction failed: %s\n", e.what());
    return 3;
  }

  p->nspin = nspin;
  p->params = impl.release();
  p->params_size = sizeof(ShimImpl);
  fill_dimensions(&p->dim, family, nspin, needs_lapl);
  return 0;
}

extern "C" void xc_func_end(xc_func_type *p) {
  if(!p) return;
  auto *impl = get_impl(p);
  if(impl) {
    delete impl;
    p->params = nullptr;
  }
}

extern "C" void xc_func_free(xc_func_type *p) {
  if(!p) return;
  // ensure end was called even if user forgot
  xc_func_end(p);
  std::free(p);
}

static inline int detect_order_lda (const xc_lda_out_params*  o){ if(o->v4rho4) return 4; if(o->v3rho3) return 3; if(o->v2rho2) return 2; if(o->vrho) return 1; if(o->zk) return 0; return -1; }
static inline int detect_order_gga (const xc_gga_out_params*  o){ if(o->v4sigma4) return 4; if(o->v3sigma3) return 3; if(o->v2sigma2) return 2; if(o->vsigma||o->vrho) return 1; if(o->zk) return 0; return -1; }
static inline int detect_order_mgga(const xc_mgga_out_params* o){ if(o->v4tau4) return 4; if(o->v3tau3) return 3; if(o->v2tau2) return 2; if(o->vtau||o->vlapl||o->vsigma||o->vrho) return 1; if(o->zk) return 0; return -1; }

/* ---------------- Device entry points (AoS on device) ---------------- */
/* NOTE: orders >= 2 return 2 (not implemented yet) */

extern "C" int GDFT_xc_lda(
  cudaStream_t stream_v,
  const xc_func_type *func, int np, const double *rho,
  xc_lda_out_params *out, xc_lda_out_params* /*buf*/
){
  if(!func || !rho || !out || np <= 0) return 1;
  auto* impl = get_impl(func);
  if(!impl || !impl->k) return 1;

  const int order = detect_order_lda(out);
  if(order < 0) return 0;
  if(order > 1){
    std::fprintf(stderr, "ExchCXX device: LDA order %d not implemented\n", order);
    return 2;
  }

  auto stream = reinterpret_cast<cudaStream_t>(stream_v);
  double* eps_dev  = out->zk;
  double* vrho_dev = out->vrho;

  if(order == 0){
    impl->k->eval_exc_vxc_device(np, rho, /*sigma*/nullptr,
                                 eps_dev, /*vrho*/nullptr, /*vsigma*/nullptr, stream);
  } else {
    impl->k->eval_exc_vxc_device(np, rho, /*sigma*/nullptr,
                                 eps_dev, vrho_dev, /*vsigma*/nullptr, stream);
  }
}

extern "C" int GDFT_xc_gga(
  cudaStream_t stream_v,
  const xc_func_type *func, int np, const double *rho, const double *sigma,
  xc_gga_out_params *out, xc_gga_out_params* /*buf*/
){
  if(!func || !rho || !out || np <= 0) return 1;
  auto* impl = get_impl(func);
  if(!impl || !impl->k) return 1;

  const int order = detect_order_gga(out);
  if(order < 0) return 0;
  if(order > 1){
    std::fprintf(stderr, "ExchCXX device: GGA order %d not implemented\n", order);
    return 2;
  }

  auto stream = reinterpret_cast<cudaStream_t>(stream_v);
  double* eps_dev    = out->zk;
  double* vrho_dev   = out->vrho;
  double* vsigma_dev = out->vsigma;

  if(order == 0){
    impl->k->eval_exc_vxc_device(np, rho, sigma,
                                 eps_dev, /*vrho*/nullptr, /*vsigma*/nullptr, stream);
  } else {
    impl->k->eval_exc_vxc_device(np, rho, sigma,
                                 eps_dev, vrho_dev, vsigma_dev, stream);
  }
}

extern "C" int GDFT_xc_mgga(
  cudaStream_t stream_v,
  const xc_func_type *func, int np,
  const double *rho, const double *sigma, const double *lapl, const double *tau,
  xc_mgga_out_params *out, xc_mgga_out_params* /*buf*/
){
  if(!func || !rho || !out || np <= 0) return 1;
  auto* impl = get_impl(func);
  if(!impl || !impl->k) return 1;

  const int order = detect_order_mgga(out);
  if(order < 0) return 0;
  if(order > 1){
    std::fprintf(stderr, "ExchCXX device: mGGA order %d not implemented\n", order);
    return 2;
  }

  auto stream = reinterpret_cast<cudaStream_t>(stream_v);
  const double* lapl_arg = (out->vlapl != nullptr) ? lapl : nullptr;

  double* eps_dev   = out->zk;
  double* vrho_dev  = out->vrho;
  double* vsig_dev  = out->vsigma;
  double* vlapl_dev = out->vlapl;  // may be null
  double* vtau_dev  = out->vtau;

  if(order == 0){
    impl->k->eval_exc_vxc_device(np, rho, sigma,
                                 eps_dev,
                                 /*vrho*/nullptr, /*vsigma*/nullptr,
                                 /*vlapl*/nullptr, /*vtau*/nullptr,
                                 lapl_arg, tau, stream);
  } else {
    impl->k->eval_exc_vxc_device(np, rho, sigma,
                                 eps_dev, vrho_dev, vsig_dev,
                                 vlapl_dev, vtau_dev,
                                 lapl_arg, tau, stream);
  }
}
