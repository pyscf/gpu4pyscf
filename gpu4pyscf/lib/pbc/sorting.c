#include <stdint.h>
#include <limits.h>
#include <omp.h>

void condense_primitive_ovlp_mask(int8_t *c_ovlp_mask, int8_t *p_ovlp_mask,
                                  int *p2c_mapping, int c_nbas, int p_nbas)
{
        for (int i = 0; i < p_nbas; i++) { int ic = p2c_mapping[i];
        for (int j = 0; j < p_nbas; j++) { int jc = p2c_mapping[j];
                c_ovlp_mask[ic*c_nbas+jc] |= p_ovlp_mask[i*p_nbas+j];
        } }
}

// out[:,idx] += inp
void take2d_add(double *out, double *inp, int *idx, int nrow, int ncol, int idxlen)
{
        if (((int64_t) nrow) * ((int64_t) ncol) < ((int64_t) INT_MAX)) {
#pragma omp parallel for schedule(static)
                for (int i = 0; i < nrow; i++) {
                for (int j = 0; j < idxlen; j++) {
                        int jp = idx[j];
                        out[i * ncol + jp] += inp[i*idxlen+j];
                } }
        } else {
                int64_t _nrow = (int64_t) nrow;
                int64_t _ncol = (int64_t) ncol;
                int64_t _idxlen = (int64_t) idxlen;
#pragma omp parallel for schedule(static)
                for (int64_t i = 0; i < _nrow; i++) {
                for (int64_t j = 0; j < _idxlen; j++) {
                        int64_t jp = idx[j];
                        out[i * _ncol + jp] += inp[i * _idxlen + j];
                } }
        }
}
