#pragma once
typedef struct {
    uint8_t li;
    uint8_t lj;
    uint8_t lk;
    uint8_t ll;
    int npairs_ij;
    int npairs_kl;
    int *pair_ij_mapping; // the significant ij pairs, mapping to i*nao+j
    int *pair_kl_mapping;
    int *pair_ij_loc; // offsets to the input dm_xyz for each ij pair
    int *pair_kl_loc;
    float *qd_ij_max; // largest dm_cond*q_cond within each block for ij pair
    float *qd_kl_max;
    float *q_cond;
    float q_cutoff; // cutoff to screening schwarz estimation q_ij+q_kl
    float qd_cutoff; // cutoff to screening contribution to J matrix
} MDBoundsInfo;
