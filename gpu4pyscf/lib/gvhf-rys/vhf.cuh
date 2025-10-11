#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#define PTR_RANGE_OMEGA 8
// slots of atm
#define CHARGE_OF       0
#define PTR_COORD       1
#define NUC_MOD_OF      2
#define PTR_ZETA        3
#define PTR_FRAC_CHARGE 4
#define RESERVE_ATMSLOT 5
#define ATM_SLOTS       6

// slots of bas
#define ATOM_OF         0
#define ANG_OF          1
#define NPRIM_OF        2
#define NCTR_OF         3
#define KAPPA_OF        4
#define PTR_EXP         5
#define PTR_COEFF       6
#define PTR_BAS_COORD   7
#define BAS_SLOTS       8

#define LMAX            4
#define LMAX1           (LMAX+1)
#define NCART_MAX       ((LMAX+1)*(LMAX+2)/2)

// performance drop when TILE>2, reason unclear
#define TILE            2
#define TILE2           (TILE*TILE)
#define TILE4           (TILE2*TILE2)
// when nroots > 5, GWIDTH=57 may be better
#define GWIDTH          42
// 2MB per block
#define QUEUE_DEPTH     262144
#define TILES_IN_BATCH  (QUEUE_DEPTH/(TILE*TILE*TILE*TILE))
#define QUEUE_DEPTH1    65536

#define MIN(x, y)       ((x) < (y) ? (x) : (y))
#define MAX(x, y)       ((x) > (y) ? (x) : (y))

// 2*pi**2.5
#define PI_FAC          34.98683665524972497


#pragma once
typedef struct {
    int natm;
    int nbas;
    int *atm;
    int *bas;
    double *env;
    int *ao_loc;
} RysIntEnvVars;

typedef struct {
    double *vj;
    double *vk;
    double *dm;
    int n_dm;
    int atom_offset;
    double omega;
    double lr_factor; // Long-range part of HF exchange
    double sr_factor; // Song-range part of HF exchange
} JKMatrix;

typedef struct {
    double *ejk;
    double *dm;
    double j_factor;
    double k_factor;
    int n_dm;
    double omega;
    double lr_factor;
    double sr_factor;
} JKEnergy;

typedef struct {
    int li;
    int lj;
    int lk;
    int ll;
    int nfi;
    int nfj;
    int nfk;
    int nfl;
    int nroots;
    int stride_j;
    int stride_k;
    int stride_l;
    int g_size;
    int iprim;
    int jprim;
    int kprim;
    int lprim;
    int npairs_ij;
    int npairs_kl;
    uint32_t *pair_ij_mapping;
    uint32_t *pair_kl_mapping;
    float *q_cond;
    float *s_estimator;
    float *dm_cond;
    float cutoff;
    int ntiles_i;
    int ntiles_j;
    int ntiles_k;
    int ntiles_l;
} BoundsInfo;

typedef struct {
    int8_t ioff;
    int8_t joff;
    int8_t koff;
    int8_t loff;
} GXYZOffset;

typedef struct {
    uint16_t i;
    uint16_t j;
    uint16_t k;
    uint16_t l;
} ShellQuartet;

typedef struct {
    uint8_t x;
    uint8_t y;
    uint16_t fold3offset;
} Fold2Index;

typedef struct {
    uint8_t x;
    uint8_t y;
    uint8_t z;
    uint8_t fold2yz;
} Fold3Index;

#ifdef __CUDACC__
__device__ __forceinline__ unsigned get_smid()
{
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

// to ensure that each SM only executes one block
#define adjust_threads(kernel, threads) { \
    cudaFuncAttributes attr; \
    cudaFuncGetAttributes(&attr, kernel); \
    if (attr.numRegs <= 128) threads *= 2; }

extern __constant__ Fold2Index c_i_in_fold2idx[];
extern __constant__ Fold3Index c_i_in_fold3idx[];

extern __constant__ int _c_cartesian_lexical_xyz[];
extern __constant__ GXYZOffset c_gxyz_offset[];
#endif
