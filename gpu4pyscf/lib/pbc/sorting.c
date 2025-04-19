#include <stdint.h>

void condense_primitive_ovlp_mask(int8_t *c_ovlp_mask, int8_t *p_ovlp_mask,
                                  int *p2c_mapping, int c_nbas, int p_nbas)
{
        for (int i = 0; i < p_nbas; i++) { int ic = p2c_mapping[i];
        for (int j = 0; j < p_nbas; j++) { int jc = p2c_mapping[j];
                c_ovlp_mask[ic*c_nbas+jc] |= p_ovlp_mask[i*p_nbas+j];
        } }
}
