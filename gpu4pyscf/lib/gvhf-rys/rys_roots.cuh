#define MAX_ORDER       9
#define DEGREE          13
#define DEGREE1         (DEGREE+1)
#define INTERVAL        2.5
#define INTERVALS       40

#include "gpu_compat.h"

#ifdef USE_SYCL
#include "rys_roots_dat.cu"
#else
extern DEVICE_CONST double ROOT_SMALLX_R0[];
extern DEVICE_CONST double ROOT_SMALLX_R1[];
extern DEVICE_CONST double ROOT_SMALLX_W0[];
extern DEVICE_CONST double ROOT_SMALLX_W1[];
extern DEVICE_CONST double ROOT_LARGEX_R_DATA[];
extern DEVICE_CONST double ROOT_LARGEX_W_DATA[];
extern DEVICE_CONST double ROOT_RW_DATA[];
#endif
