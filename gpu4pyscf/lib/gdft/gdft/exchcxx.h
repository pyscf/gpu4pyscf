/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifndef _XC_H
#define _XC_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Version / reference (minimal ABI) ---- */
const char *xc_reference(void);
const char *xc_reference_doi(void);
const char *xc_reference_key(void);
void xc_version(int *major, int *minor, int *micro);
const char *xc_version_string(void);

/* ---- Common constants (kept identical to original) ---- */
#include <stddef.h>

#define XC_UNPOLARIZED          1
#define XC_POLARIZED            2

#define XC_NON_RELATIVISTIC     0
#define XC_RELATIVISTIC         1

#define XC_EXCHANGE             0
#define XC_CORRELATION          1
#define XC_EXCHANGE_CORRELATION 2
#define XC_KINETIC              3

#define XC_FAMILY_UNKNOWN      -1
#define XC_FAMILY_LDA           1
#define XC_FAMILY_GGA           2
#define XC_FAMILY_MGGA          4
#define XC_FAMILY_LCA           8
#define XC_FAMILY_OEP          16
#define XC_FAMILY_HYB_GGA      32
#define XC_FAMILY_HYB_MGGA     64
#define XC_FAMILY_HYB_LDA     128

#define XC_FLAGS_HAVE_EXC         (1 <<  0)
#define XC_FLAGS_HAVE_VXC         (1 <<  1)
#define XC_FLAGS_HAVE_FXC         (1 <<  2)
#define XC_FLAGS_HAVE_KXC         (1 <<  3)
#define XC_FLAGS_HAVE_LXC         (1 <<  4)
#define XC_FLAGS_1D               (1 <<  5)
#define XC_FLAGS_2D               (1 <<  6)
#define XC_FLAGS_3D               (1 <<  7)
#define XC_FLAGS_HYB_CAM          (1 <<  8)
#define XC_FLAGS_HYB_CAMY         (1 <<  9)
#define XC_FLAGS_VV10             (1 << 10)
#define XC_FLAGS_HYB_LC           (1 << 11)
#define XC_FLAGS_HYB_LCY          (1 << 12)
#define XC_FLAGS_STABLE           (1 << 13)
#define XC_FLAGS_DEVELOPMENT      (1 << 14)
#define XC_FLAGS_NEEDS_LAPLACIAN  (1 << 15)
#define XC_FLAGS_NEEDS_TAU        (1 << 16)
#define XC_FLAGS_HAVE_ALL         (XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC | XC_FLAGS_HAVE_LXC)

#define XC_EXT_PARAMS_DEFAULT   -999998888
#define XC_MAX_REFERENCES       5

/* ---- Output param macros kept for ABI (unused in this shim) ---- */
#define XC_NOARG
#define XC_COMMA ,

#define LDA_OUT_PARAMS_NO_EXC(P1_, P2_) \
  P1_ P2_ ## vrho   \
  P1_ P2_ ## v2rho2 \
  P1_ P2_ ## v3rho3 \
  P1_ P2_ ## v4rho4

#define GGA_OUT_PARAMS_NO_EXC(P1_, P2_) \
  P1_ P2_ ## vrho         P1_ P2_ ## vsigma       \
  P1_ P2_ ## v2rho2       P1_ P2_ ## v2rhosigma   \
  P1_ P2_ ## v2sigma2                             \
  P1_ P2_ ## v3rho3       P1_ P2_ ## v3rho2sigma  \
  P1_ P2_ ## v3rhosigma2  P1_ P2_ ## v3sigma3     \
  P1_ P2_ ## v4rho4       P1_ P2_ ## v4rho3sigma  \
  P1_ P2_ ## v4rho2sigma2 P1_ P2_ ## v4rhosigma3  \
  P1_ P2_ ## v4sigma4

#define MGGA_OUT_PARAMS_NO_EXC(P1_, P2_) \
  P1_ P2_ ## vrho              P1_ P2_ ## vsigma          \
  P1_ P2_ ## vlapl             P1_ P2_ ## vtau            \
  P1_ P2_ ## v2rho2            P1_ P2_ ## v2rhosigma      \
  P1_ P2_ ## v2rholapl         P1_ P2_ ## v2rhotau        \
  P1_ P2_ ## v2sigma2          P1_ P2_ ## v2sigmalapl     \
  P1_ P2_ ## v2sigmatau        P1_ P2_ ## v2lapl2         \
  P1_ P2_ ## v2lapltau         P1_ P2_ ## v2tau2          \
  P1_ P2_ ## v3rho3            P1_ P2_ ## v3rho2sigma     \
  P1_ P2_ ## v3rho2lapl        P1_ P2_ ## v3rho2tau       \
  P1_ P2_ ## v3rhosigma2       P1_ P2_ ## v3rhosigmalapl  \
  P1_ P2_ ## v3rhosigmatau     P1_ P2_ ## v3rholapl2      \
  P1_ P2_ ## v3rholapltau      P1_ P2_ ## v3rhotau2       \
  P1_ P2_ ## v3sigma3          P1_ P2_ ## v3sigma2lapl    \
  P1_ P2_ ## v3sigma2tau       P1_ P2_ ## v3sigmalapl2    \
  P1_ P2_ ## v3sigmalapltau    P1_ P2_ ## v3sigmatau2     \
  P1_ P2_ ## v3lapl3           P1_ P2_ ## v3lapl2tau      \
  P1_ P2_ ## v3lapltau2        P1_ P2_ ## v3tau3          \
  P1_ P2_ ## v4rho4            P1_ P2_ ## v4rho3sigma     \
  P1_ P2_ ## v4rho3lapl        P1_ P2_ ## v4rho3tau       \
  P1_ P2_ ## v4rho2sigma2      P1_ P2_ ## v4rho2sigmalapl \
  P1_ P2_ ## v4rho2sigmatau    P1_ P2_ ## v4rho2lapl2     \
  P1_ P2_ ## v4rho2lapltau     P1_ P2_ ## v4rho2tau2      \
  P1_ P2_ ## v4rhosigma3       P1_ P2_ ## v4rhosigma2lapl \
  P1_ P2_ ## v4rhosigma2tau    P1_ P2_ ## v4rhosigmalapl2 \
  P1_ P2_ ## v4rhosigmalapltau P1_ P2_ ## v4rhosigmatau2  \
  P1_ P2_ ## v4rholapl3        P1_ P2_ ## v4rholapl2tau   \
  P1_ P2_ ## v4rholapltau2     P1_ P2_ ## v4rhotau3       \
  P1_ P2_ ## v4sigma4          P1_ P2_ ## v4sigma3lapl    \
  P1_ P2_ ## v4sigma3tau       P1_ P2_ ## v4sigma2lapl2   \
  P1_ P2_ ## v4sigma2lapltau   P1_ P2_ ## v4sigma2tau2    \
  P1_ P2_ ## v4sigmalapl3      P1_ P2_ ## v4sigmalapl2tau \
  P1_ P2_ ## v4sigmalapltau2   P1_ P2_ ## v4sigmatau3     \
  P1_ P2_ ## v4lapl4           P1_ P2_ ## v4lapl3tau      \
  P1_ P2_ ## v4lapl2tau2       P1_ P2_ ## v4lapltau3      \
  P1_ P2_ ## v4tau4

/* ---- C structs (match your Python ctypes exactly) ---- */
typedef struct{
  const char *ref, *doi, *bibtex, *key;
} func_reference_type;

typedef struct{
  int n;
  const char **names;
  const char **descriptions;
  const double *values;
  void (*set)(struct xc_func_type *p, const double *ext_params);
} func_params_type;

typedef struct {
  int rho, sigma, lapl, tau;
  int zk MGGA_OUT_PARAMS_NO_EXC(XC_COMMA, );
} xc_dimensions;

typedef struct xc_func_info_type {
  int   number;
  int   kind;
  const char *name;
  int   family;
  func_reference_type *refs[XC_MAX_REFERENCES];
  int   flags;
  double dens_threshold;
  func_params_type ext_params;
  void (*init)(struct xc_func_type *p);
  void (*end) (struct xc_func_type *p);
  const void *lda;   /* unused by shim */
  const void *gga;   /* unused by shim */
  const void *mgga;  /* unused by shim */
} xc_func_info_type;

typedef struct xc_func_type{
  const xc_func_info_type *info;
  int nspin;
  int n_func_aux;
  struct xc_func_type **func_aux;
  double *mix_coef;

  double cam_omega, cam_alpha, cam_beta;
  double nlc_b, nlc_C;

  xc_dimensions dim;

  double *ext_params;
  void *params;      /* shim stores internal ExchCXX state here */
  int params_size;

  double dens_threshold;
  double zeta_threshold;
  double sigma_threshold;
  double tau_threshold;
} xc_func_type;

/* ---- Output parameter containers ---- */
typedef struct {
  double *zk;
  double *vrho;
  double *v2rho2;
  double *v3rho3;
  double *v4rho4;
} xc_lda_out_params;

typedef struct {
  double *zk;
  double *vrho, *vsigma;
  double *v2rho2, *v2rhosigma, *v2sigma2;
  double *v3rho3, *v3rho2sigma, *v3rhosigma2, *v3sigma3;
  double *v4rho4, *v4rho3sigma, *v4rho2sigma2, *v4rhosigma3, *v4sigma4;
} xc_gga_out_params;

typedef struct {
  double *zk;
  double *vrho, *vsigma, *vlapl, *vtau;
  double *v2rho2, *v2rhosigma, *v2rholapl, *v2rhotau, *v2sigma2;
  double *v2sigmalapl, *v2sigmatau, *v2lapl2, *v2lapltau, *v2tau2;
  double *v3rho3, *v3rho2sigma, *v3rho2lapl, *v3rho2tau, *v3rhosigma2;
  double *v3rhosigmalapl, *v3rhosigmatau, *v3rholapl2, *v3rholapltau;
  double *v3rhotau2, *v3sigma3, *v3sigma2lapl, *v3sigma2tau;
  double *v3sigmalapl2, *v3sigmalapltau, *v3sigmatau2, *v3lapl3;
  double *v3lapl2tau, *v3lapltau2, *v3tau3;
  double *v4rho4, *v4rho3sigma, *v4rho3lapl, *v4rho3tau, *v4rho2sigma2;
  double *v4rho2sigmalapl, *v4rho2sigmatau, *v4rho2lapl2, *v4rho2lapltau;
  double *v4rho2tau2, *v4rhosigma3, *v4rhosigma2lapl, *v4rhosigma2tau;
  double *v4rhosigmalapl2, *v4rhosigmalapltau,  *v4rhosigmatau2;
  double *v4rholapl3, *v4rholapl2tau, *v4rholapltau2, *v4rhotau3;
  double *v4sigma4, *v4sigma3lapl, *v4sigma3tau, *v4sigma2lapl2;
  double *v4sigma2lapltau, *v4sigma2tau2, *v4sigmalapl3, *v4sigmalapl2tau;
  double *v4sigmalapltau2, *v4sigmatau3, *v4lapl4, *v4lapl3tau;
  double *v4lapl2tau2, *v4lapltau3, *v4tau4;
} xc_mgga_out_params;

/* ---- Minimal API we implement ---- */
xc_func_type *xc_func_alloc(void);
int   xc_func_init(xc_func_type *p, int functional, int nspin);
void  xc_func_end(xc_func_type *p);
void  xc_func_free(xc_func_type *p);

/* String <-> id helper (we implement a small table; extend as needed) */
int   xc_functional_get_number(const char *name);

/* ---- Device entry points used by Python (unchanged ABI) ---- */
int GDFT_xc_lda (void* stream,
  const xc_func_type *func, int np, const double *rho,
  xc_lda_out_params *out, xc_lda_out_params *buf);

int GDFT_xc_gga (void* stream,
  const xc_func_type *func, int np, const double *rho, const double *sigma,
  xc_gga_out_params *out, xc_gga_out_params *buf);

int GDFT_xc_mgga(void* stream,
  const xc_func_type *func, int np,
  const double *rho, const double *sigma, const double *lapl, const double *tau,
  xc_mgga_out_params *out, xc_mgga_out_params *buf);

#ifdef __cplusplus
}
#endif
#endif /* _XC_H */
