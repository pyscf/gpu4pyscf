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


#ifndef HAVE_DEFINED_INTENVVAS_H
#define HAVE_DEFINED_INTENVVAS_H
typedef struct {
    uint16_t natm;
    uint16_t nbas;
    int *atm;
    int *bas;
    double *env;
    int *ao_loc;
} RysIntEnvVars;

typedef struct {
    double *vj;
    double *vk;
    double *dm;
    uint16_t n_dm;
    uint16_t atom_offset;
} JKMatrix;

typedef struct {
    double *ejk;
    double *dm;
    double j_factor;
    double k_factor;
    uint16_t n_dm;
} JKEnergy;

typedef struct {
    uint8_t li;
    uint8_t lj;
    uint8_t lk;
    uint8_t ll;
    uint8_t nfi;
    uint8_t nfk;
    uint8_t nfij;
    uint8_t nfkl;
    uint8_t nroots;
    uint8_t stride_j;
    uint8_t stride_k;
    uint8_t stride_l;
    uint8_t iprim;
    uint8_t jprim;
    uint8_t kprim;
    uint8_t lprim;
    union {int ntile_ij_pairs; int npairs_ij;};
    union {int ntile_kl_pairs; int npairs_kl;};
    union {int *tile_ij_mapping; int *pair_ij_mapping;};
    union {int *tile_kl_mapping; int *pair_kl_mapping;};
    float *q_cond;
    float *tile_q_cond;
    float *s_estimator;
    float *dm_cond;
    float cutoff;
} BoundsInfo;

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
#endif

#ifdef __CUDACC__
extern __constant__ int c_g_pair_idx[];
extern __constant__ int c_g_pair_offsets[];
//extern __constant__ double c_env[];
extern __constant__ Fold2Index c_i_in_fold2idx[];
extern __constant__ Fold3Index c_i_in_fold3idx[];
#endif
