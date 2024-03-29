from . import mp2
from . import dfmp2

def MP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if mf.istype('UHF'):
        raise NotImplementedError
        #return UMP2(mf, frozen, mo_coeff, mo_occ)
    elif mf.istype('GHF'):
        raise NotImplementedError
        #return GMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return RMP2(mf, frozen, mo_coeff, mo_occ)

def RMP2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    from pyscf import lib

    if mf.istype('UHF'):
        raise RuntimeError('RMP2 cannot be used with UHF method.')
    elif mf.istype('ROHF'):
        raise NotImplementedError
        lib.logger.warn(mf, 'RMP2 method does not support ROHF method. ROHF object '
                        'is converted to UHF object and UMP2 method is called.')
        #return UMP2(mf, frozen, mo_coeff, mo_occ)

    #mf = mf.remove_soscf()
    if not mf.istype('RHF'):
        mf = mf.to_rhf()

    if getattr(mf, 'with_df', None):
        return dfmp2.DFMP2(mf, frozen, mo_coeff, mo_occ)
    else:
        return mp2.RMP2(mf, frozen, mo_coeff, mo_occ)