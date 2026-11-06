# cupyx.scipy.spatial.distance shim for the SYCL (dpnp) backend.
import dpnp


def cdist(XA, XB, metric='euclidean', **kwargs):
    '''Pairwise distances between the rows of XA (m, n) and XB (p, n).'''
    if metric not in ('euclidean', 'sqeuclidean'):
        raise NotImplementedError(f'cdist metric {metric!r}')
    XA = dpnp.asarray(XA, dtype=dpnp.float64)
    XB = dpnp.asarray(XB, dtype=dpnp.float64)
    d2 = ((XA[:, None, :] - XB[None, :, :])**2).sum(axis=-1)
    return d2 if metric == 'sqeuclidean' else dpnp.sqrt(d2)
