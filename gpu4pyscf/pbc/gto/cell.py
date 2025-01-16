import numpy as np

# This function is only available in pyscf-2.8 or later
def extract_pgto_params(cell, op='diffused'):
    '''A helper function to extract exponents and contraction coefficients for
    estimate_xxx function
    '''
    es = []
    cs = []
    if op == 'diffused':
        precision = cell.precision
        for i in range(cell.nbas):
            e = cell.bas_exp(i)
            c = abs(cell._libcint_ctr_coeff(i)).max(axis=1)
            l = cell.bas_angular(i)
            # A quick estimation for the radius that each primitive GTO vanishes
            r2 = np.log(c**2 / precision * 10**l) / e
            idx = r2.argmax()
            es.append(e[idx])
            cs.append(c[idx].max())
    elif op == 'compact':
        precision = cell.precision
        for i in range(cell.nbas):
            e = cell.bas_exp(i)
            c = abs(cell._libcint_ctr_coeff(i)).max(axis=1)
            l = cell.bas_angular(i)
            # A quick estimation for the resolution of planewaves that each
            # primitive GTO requires
            ke = np.log(c**2 / precision * 50**l) * e
            idx = ke.argmax()
            es.append(e[idx])
            cs.append(c[idx].max())
    else:
        raise RuntimeError(f'Unsupported operation {op}')
    return np.array(es), np.array(cs)
