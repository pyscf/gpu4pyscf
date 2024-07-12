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

/* Get the literature reference for libxc */
const char *xc_reference();
/* Get the doi for the literature reference for libxc */
const char *xc_reference_doi();
/* Get the key for the literature reference for libxc */
const char *xc_reference_key();

/* Get the major, minor, and micro version of libxc */
void xc_version(int *major, int *minor, int *micro);
/* Get the version of libxc as a string */
const char *xc_version_string();

//#include <xc_version.h>
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

/* flags that can be used in info.flags. Don't reorder these since it
   will break the ABI of the library. */
#define XC_FLAGS_HAVE_EXC         (1 <<  0) /*     1 */
#define XC_FLAGS_HAVE_VXC         (1 <<  1) /*     2 */
#define XC_FLAGS_HAVE_FXC         (1 <<  2) /*     4 */
#define XC_FLAGS_HAVE_KXC         (1 <<  3) /*     8 */
#define XC_FLAGS_HAVE_LXC         (1 <<  4) /*    16 */
#define XC_FLAGS_1D               (1 <<  5) /*    32 */
#define XC_FLAGS_2D               (1 <<  6) /*    64 */
#define XC_FLAGS_3D               (1 <<  7) /*   128 */
#define XC_FLAGS_VV10             (1 << 10) /*  1024 */
#define XC_FLAGS_STABLE           (1 << 13) /*  8192 */
/* functionals marked with the development flag may have significant problems in the implementation */
#define XC_FLAGS_DEVELOPMENT      (1 << 14) /* 16384 */
#define XC_FLAGS_NEEDS_LAPLACIAN  (1 << 15) /* 32768 */
#define XC_FLAGS_NEEDS_TAU        (1 << 16) /* 65536 */

/* This is the case for most functionals in libxc */
#define XC_FLAGS_HAVE_ALL         (XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | \
                                   XC_FLAGS_HAVE_FXC | XC_FLAGS_HAVE_KXC | \
                                   XC_FLAGS_HAVE_LXC)

/* This magic value means use default parameter */
#define XC_EXT_PARAMS_DEFAULT   -999998888

/* Different flavors of many-body terms used in hybrids
   The Fock term to be added to the Hamiltonian reads

     F = -1/2 <i j| f(r_12)/r_12 |j i>

   where the function f(r) is

   *) XC_HYB_FOCK           f(r) = coeff
   *) XC_HYB_ERF_SR         f(r) = coeff * (1 - erf(omega r))
   *) XC_HYB_YUKAWA_SR      f(r) = coeff * exp(-omega r)
   *) XC_HYB_GAUSSIAN_SR    f(r) = coeff * 2*omega/sqrt(pi) * exp(-omega^2 r^2)
*/
#define XC_HYB_NONE             0
#define XC_HYB_FOCK             1  /* Normal hybrid */
#define XC_HYB_PT2              2  /* Used for double hybrids */
#define XC_HYB_ERF_SR           4  /* Short range of range separated - erf version */
#define XC_HYB_YUKAWA_SR        8  /* Short range of range separated - Yakawa version */
#define XC_HYB_GAUSSIAN_SR     16  /* Short range of range separated - Gaussian version */

/* Different types of hybrid functionals. */
#define XC_HYB_SEMILOCAL        0  /* Standard semi-local functional (not a hybrid) */
#define XC_HYB_HYBRID           1  /* Standard hybrid functional */
#define XC_HYB_CAM              2  /* Coulomb attenuated hybrid */
#define XC_HYB_CAMY             3  /* Coulomb attenuated hybrid with a Yukawa screening */
#define XC_HYB_CAMG             4  /* Coulomb attenuated hybrid with a Gaussian screening */
#define XC_HYB_DOUBLE_HYBRID    5  /* Double hybrid */
#define XC_HYB_MIXTURE      32768  /* More complicated mixture (have to check individual terms) */

#define XC_MAX_REFERENCES       5

/* This are the derivatives that a functional returns */
#define XC_NOARG
#define XC_COMMA ,

/* the following macros *do not* include zk */
/* the following macros are probably to DELETE */

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

/* This are the derivatives of a mgga
       1st order:  4
       2nd order: 10
       3rd order: 20
       4th order: 35
 */
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


struct xc_func_type;

typedef struct{
  const char *ref, *doi, *bibtex, *key;
} func_reference_type;

const char *xc_func_reference_get_ref(const func_reference_type *reference);
const char *xc_func_reference_get_doi(const func_reference_type *reference);
const char *xc_func_reference_get_bibtex(const func_reference_type *reference);
const char *xc_func_reference_get_key(const func_reference_type *reference);


typedef struct{
  int n; /* Number of parameters */

  const char **names; /* ATTENTION: if name starts with a _ it is an *internal* parameter,
                        changing the value effectively changes the functional! */
  const char **descriptions; /* long description of the parameters */
  const double *values; /* default values of the parameters */

  void (*set)(struct xc_func_type *p, const double *ext_params);
} func_params_type;


/* In the future these following three structures might be unified */
typedef struct {
  /* order 0 */
  double *zk;
  /* order 1 */
  double *vrho;
  /* order 2 */
  double *v2rho2;
  /* order 3 */
  double *v3rho3;
  /* order 4 */
  double *v4rho4;
} xc_lda_out_params;

typedef struct {
  /* order 0 */
  double *zk;
  /* order 1 */
  double *vrho, *vsigma;
  /* order 2 */
  double *v2rho2, *v2rhosigma, *v2sigma2;
  /* order 3 */
  double *v3rho3, *v3rho2sigma, *v3rhosigma2, *v3sigma3;
  /* order 4 */
  double *v4rho4, *v4rho3sigma, *v4rho2sigma2, *v4rhosigma3, *v4sigma4;
} xc_gga_out_params;

typedef struct {
  /* order 0 */
  double *zk;
  /* order 1 */
  double *vrho, *vsigma, *vlapl, *vtau;
  /* order 2 */
  double *v2rho2, *v2rhosigma, *v2rholapl, *v2rhotau, *v2sigma2;
  double *v2sigmalapl, *v2sigmatau, *v2lapl2, *v2lapltau, *v2tau2;
  /* order 3 */
  double *v3rho3, *v3rho2sigma, *v3rho2lapl, *v3rho2tau, *v3rhosigma2;
  double *v3rhosigmalapl, *v3rhosigmatau, *v3rholapl2, *v3rholapltau;
  double *v3rhotau2, *v3sigma3, *v3sigma2lapl, *v3sigma2tau;
  double *v3sigmalapl2, *v3sigmalapltau, *v3sigmatau2, *v3lapl3;
  double *v3lapl2tau, *v3lapltau2, *v3tau3;
  /* order 4 */
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

/* type of the lda function */
typedef void (*xc_lda_funcs)
(const struct xc_func_type *p, size_t np,
 const double *rho,
 xc_lda_out_params *out);

typedef struct {
  const xc_lda_funcs unpol[5], pol[5];
} xc_lda_funcs_variants;

/* type of the gga function */
typedef void (*xc_gga_funcs)
(const struct xc_func_type *p, size_t np,
 const double *rho, const double *sigma,
 xc_gga_out_params *out);

typedef struct {
  const xc_gga_funcs unpol[5], pol[5];
} xc_gga_funcs_variants;

/* type of the mgga function */
typedef void (*xc_mgga_funcs)
(const struct xc_func_type *p, size_t np,
 const double *rho, const double *sigma, const double *lapl, const double *tau,
 xc_mgga_out_params *out);
typedef struct {
  const xc_mgga_funcs unpol[5], pol[5];
} xc_mgga_funcs_variants;


typedef struct{
  int   number;   /* identifier number */
  int   kind;     /* XC_EXCHANGE, XC_CORRELATION, XC_EXCHANGE_CORRELATION, XC_KINETIC */

  const char *name;     /* name of the functional, e.g. "PBE" */
  int   family;   /* type of the functional, e.g. XC_FAMILY_GGA */
  func_reference_type *refs[XC_MAX_REFERENCES];  /* index of the references */

  int   flags;    /* see above for a list of possible flags */

  double dens_threshold;

  /* this allows to have external parameters in the functional */
  func_params_type ext_params;

  void (*init)(struct xc_func_type *p);
  void (*end) (struct xc_func_type *p);
  const xc_lda_funcs_variants  *lda;
  const xc_gga_funcs_variants  *gga;
  const xc_mgga_funcs_variants *mgga;
} xc_func_info_type;


/* for API compability with older versions of libxc */
#define XC(func) xc_ ## func


int xc_func_info_get_number(const xc_func_info_type *info);
int xc_func_info_get_kind(const xc_func_info_type *info);
char const *xc_func_info_get_name(const xc_func_info_type *info);
int xc_func_info_get_family(const xc_func_info_type *info);
int xc_func_info_get_flags(const xc_func_info_type *info);
const func_reference_type *xc_func_info_get_references(const xc_func_info_type *info, int number);


int xc_func_info_get_n_ext_params(const xc_func_info_type *info);
char const *xc_func_info_get_ext_params_name(const xc_func_info_type *p, int number);
char const *xc_func_info_get_ext_params_description(const xc_func_info_type *info, int number);
double xc_func_info_get_ext_params_default_value(const xc_func_info_type *info, int number);


struct xc_dimensions{
  int rho, sigma, lapl, tau;       /* spin dimensions of the arrays */
  int zk MGGA_OUT_PARAMS_NO_EXC(XC_COMMA, );
};

typedef struct xc_dimensions xc_dimensions;


struct xc_func_type{
  const xc_func_info_type *info;       /* all the information concerning this functional */
  int nspin;                           /* XC_UNPOLARIZED or XC_POLARIZED  */

  int n_func_aux;                      /* how many auxiliary functions we need */
  struct xc_func_type **func_aux;      /* most GGAs are based on a LDA or other GGAs  */
  double *mix_coef;                    /* coefficients for the mixing */

  /**
     Parameters for range-separated hybrids
     hyb_type[i]:  XC_HYB_NONE, XC_HYB_FOCK, XC_HYB_ERF_SR, etc.
     hyb_omega[i]: the range separation constant
     hyb_coeff[i]: fraction of exchange, used both for
                usual hybrids as well as range-separated ones

     N.B. Different conventions for alpha and beta can be found in
     literature. In the convention used in libxc, at short range the
     fraction of exact exchange is cam_alpha+cam_beta, while at long
     range it is cam_alpha.
  */
  int hyb_number_terms, *hyb_type;
  double *hyb_coeff, *hyb_omega;

  double nlc_b;                /* Non-local correlation, b parameter */
  double nlc_C;                /* Non-local correlation, C parameter */

  xc_dimensions dim;           /* the dimensions of all input and output arrays */

  /* This is where the values of the external parameters are stored */
  double *ext_params;
  /* This is a placeholder for structs of parameters that are used in the Maple generated sources */
  void *params;

  double dens_threshold;       /* functional is put to zero for spin-densities smaller than this */
  double zeta_threshold;       /* idem for the absolute value of zeta */
  double sigma_threshold;
  double tau_threshold;
};

typedef struct xc_func_type xc_func_type;


/** Get a functional's id number from its name  */
int   xc_functional_get_number(const char *name);
/** Get a functional's name from its id number  */
char *xc_functional_get_name(int number);
/** Get a functional's family and the number within the family from the id number */
int   xc_family_from_id(int id, int *family, int *number);

/** The number of functionals implemented in this version of libxc */
int   xc_number_of_functionals();
/** The maximum name length of any functional */
int   xc_maximum_name_length();
/** Returns the available functional number sorted by id */
void  xc_available_functional_numbers(int *list);
/** Returns the available functional number sorted by the functionals'
    names; this function is a helper for the Python frontend. */
void  xc_available_functional_numbers_by_name(int *list);
/** Fills the list with the names of the available functionals,
    ordered by name. The list array should be [Nfuncs][maxlen+1]. */
void  xc_available_functional_names(char **list);

/** Dynamically allocates a libxc functional; which will also need to be initialized. */
xc_func_type *xc_func_alloc();
/** Initializes a functional by id with nspin spin channels */
int   xc_func_init(xc_func_type *p, int functional, int nspin);
/** Destructor for an initialized functional */
void  xc_func_end(xc_func_type *p);
/** Frees a dynamically allocated functional */
void  xc_func_free(xc_func_type *p);
/** Get information on a functional */
const xc_func_info_type *xc_func_get_info(const xc_func_type *p);

/** Sets the density threshold for a functional */
void  xc_func_set_dens_threshold(xc_func_type *p, double t_dens);
/** Sets the spin polarization threshold for a functional */
void  xc_func_set_zeta_threshold(xc_func_type *p, double t_zeta);
/** Sets the reduced gradient threshold for a functional */
void  xc_func_set_sigma_threshold(xc_func_type *p, double t_sigma);
/** Sets the kinetic energy density threshold for a functional */
void  xc_func_set_tau_threshold(xc_func_type *p, double t_tau);

/** Sets all external parameters for a functional */
void  xc_func_set_ext_params(xc_func_type *p, const double *ext_params);
/** Gets all external parameters for a functional. Array needs to be preallocated  */
void  xc_func_get_ext_params(const xc_func_type *p, double *ext_params);
/** Sets an external parameter by name for a functional */
void  xc_func_set_ext_params_name(xc_func_type *p, const char *name, double par);
/** Gets an external parameter by name for a functional */
double xc_func_get_ext_params_name(const xc_func_type *p, const char *name);
/** Gets an external parameter by index */
double xc_func_get_ext_params_value(const xc_func_type *p, int number);

#ifdef __cplusplus
}
#endif

#endif
