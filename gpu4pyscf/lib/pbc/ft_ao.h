#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#ifndef HAVE_DEFINED_AFTENVVAS_H
#define HAVE_DEFINED_AFTENVVAS_H
typedef struct {
    uint16_t natm; // in bvk-cell
    uint16_t nbas; // in bvk-cell
    int *atm;
    int *bas;
    double *env;
    int *ao_loc; // in bvk-cell
    double *img_coords; // vectors in lattice sum
    int *img_idx; // indices of img_coords in each shell-pair
    int *img_offsets; // offset AFTIntEnvVars.img_idx for each shell-pair
} AFTIntEnvVars;

typedef struct {
    uint8_t li;
    uint8_t lj;
    uint8_t nfij;
    uint8_t g_size;
    uint8_t stride_i;
    uint8_t stride_j;
    uint8_t iprim;
    uint8_t jprim;
    int npairs_ij;
    int *ish_in_pair;
    int *jsh_in_pair;
    int ngrids;
    int ngrids_in_batch;
    double *grids;
} AFTBoundsInfo;
#endif
