"""Resident DIIS for NumPy and CuPy, single compact host sync per Gram matrix."""
import numpy as np


class ResidentDIIS:
    """Keep error vectors and amplitude history on the same backend.

    For GPU use a single small Gram-matrix transfer to CPU where the
    stabilized DIIS coefficients are determined in float64.
    """
    def __init__(self, xp, space=6, rcond=1e-12):
        if space < 1:
            raise ValueError('DIIS space must be positive')
        self.xp = xp
        self.space = int(space)
        self.rcond = float(rcond)
        self.vectors = []
        self.errors = []

    def extrapolate(self, vector, error):
        xp = self.xp
        self.vectors.append(vector.copy())
        self.errors.append(error.copy())
        if len(self.vectors) > self.space:
            self.vectors.pop(0)
            self.errors.pop(0)
        n = len(self.vectors)
        if n < 2:
            return vector
        err = xp.stack(self.errors, axis=0)
        gram = err @ err.T
        b = np.empty((n+1, n+1), dtype=np.float64)
        b[0, 0] = 0.0
        b[0, 1:] = b[1:, 0] = -1.0
        b[1:, 1:] = xp.asnumpy(gram) if xp is not np else gram
        rhs = np.zeros(n+1)
        rhs[0] = -1.
        try:
            coeff = np.linalg.lstsq(b, rhs, rcond=self.rcond)[0][1:]
        except np.linalg.LinAlgError:
            return vector
        if not np.all(np.isfinite(coeff)) or np.max(np.abs(coeff)) > 1e5:
            return vector
        # A small number of launches is acceptable for max 6 history vectors.
        out = xp.zeros_like(vector)
        for c, v in zip(coeff, self.vectors):
            out += float(c) * v
        return out


class ScalarResidentDIIS(ResidentDIIS):
    """Original reference DIIS: one GPU sync per lower Gram element."""
    def extrapolate(self, vector, error):
        xp=self.xp
        self.vectors.append(vector.copy())
        self.errors.append(error.copy())
        if len(self.vectors)>self.space:
            self.vectors.pop(0)
            self.errors.pop(0)
        n=len(self.vectors)
        if n<2:
            return vector
        b=np.empty((n+1,n+1),dtype=np.float64)
        b[0,0]=0
        b[0,1:]=b[1:,0]=-1
        for i in range(n):
            for j in range(i+1):
                x=float(xp.vdot(self.errors[i],self.errors[j]).item())
                b[i+1,j+1]=b[j+1,i+1]=x
        rhs=np.zeros(n+1)
        rhs[0]=-1
        try:
            co=np.linalg.lstsq(b,rhs,rcond=self.rcond)[0][1:]
        except np.linalg.LinAlgError:
            return vector
        if not np.all(np.isfinite(co)) or np.max(np.abs(co))>1e5:
            return vector
        output=xp.zeros_like(vector)
        for c,v in zip(co,self.vectors):
            output+=float(c)*v
        return output
