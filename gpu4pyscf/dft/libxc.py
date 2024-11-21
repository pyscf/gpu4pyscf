# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2022 Qiming Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import numpy as np
import ctypes
import ctypes.util
import cupy
import copy
from ctypes import POINTER
from pyscf import dft
from gpu4pyscf.dft.libxc_structs import xc_func_type, xc_lda_out_params, xc_gga_out_params, xc_mgga_out_params
from gpu4pyscf.lib.cupy_helper import load_library
from gpu4pyscf.dft import libxc_structs

import site
path_list = [os.path.abspath(os.path.join(__file__, '..', '..', '..'))] + site.getsitepackages()
__version__ = '6.1' # hard coded

# CPU routines
is_nlc           = dft.libxc.is_nlc
is_hybrid_xc     = dft.libxc.is_hybrid_xc
test_deriv_order = dft.libxc.test_deriv_order

for path in path_list:
    libxc_path = os.path.abspath(os.path.join(path, 'gpu4pyscf', 'lib', 'deps', 'lib'))
    try:
        _libxc = np.ctypeslib.load_library('libxc', libxc_path)
        break
    except Exception:
        _libxc = None

libgdft = load_library('libgdft')
libgdft.GDFT_xc_lda.argtypes = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    POINTER(xc_lda_out_params),
    POINTER(xc_lda_out_params))

libgdft.GDFT_xc_gga.argtypes = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    POINTER(xc_gga_out_params),
    POINTER(xc_gga_out_params))

libgdft.GDFT_xc_mgga.argtypes = (
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    POINTER(xc_mgga_out_params),
    POINTER(xc_mgga_out_params))

if _libxc is None:
    import warnings
    warnings.warn(
        "Cannot find installed libXC. DFT modules may not work.\n \
        You can install libXC by \n \
        `pip3 install gpu4pyscf-libxc-cuda11x` \n \
        OR \n \
        `pip3 install gpu4pyscf-libxc-cuda12x`"
    )

LDA_OUTPUT_LABELS = [
                "zk",       # 1, 1
                "vrho",     # 1, 2
                "v2rho2",   # 1, 3
                "v3rho3",   # 1, 4
                "v4rho4"    # 1, 5
            ]

GGA_OUTPUT_LABELS = [
                "zk",                                                               # 1, 1
                "vrho", "vsigma",                                                   # 2, 3
                "v2rho2", "v2rhosigma", "v2sigma2",                                 # 3, 6
                "v3rho3", "v3rho2sigma", "v3rhosigma2", "v3sigma3",                 # 4, 10
                "v4rho4", "v4rho3sigma", "v4rho2sigma2", "v4rhosigma3", "v4sigma4"  # 5, 15
            ]

MGGA_OUTPUT_LABELS =         [
                "zk",                                                                # 1, 1
                "vrho", "vsigma", "vlapl", "vtau",                                   # 4, 5
                "v2rho2", "v2rhosigma", "v2rholapl", "v2rhotau", "v2sigma2",         # 10, 15
                "v2sigmalapl", "v2sigmatau", "v2lapl2", "v2lapltau",  "v2tau2",
                "v3rho3", "v3rho2sigma", "v3rho2lapl", "v3rho2tau", "v3rhosigma2",   # 20, 35
                "v3rhosigmalapl", "v3rhosigmatau", "v3rholapl2", "v3rholapltau",
                "v3rhotau2", "v3sigma3", "v3sigma2lapl", "v3sigma2tau",
                "v3sigmalapl2", "v3sigmalapltau", "v3sigmatau2", "v3lapl3",
                "v3lapl2tau", "v3lapltau2", "v3tau3",
                "v4rho4", "v4rho3sigma", "v4rho3lapl", "v4rho3tau", "v4rho2sigma2",  # 35, 70
                "v4rho2sigmalapl", "v4rho2sigmatau", "v4rho2lapl2", "v4rho2lapltau",
                "v4rho2tau2", "v4rhosigma3", "v4rhosigma2lapl", "v4rhosigma2tau",
                "v4rhosigmalapl2", "v4rhosigmalapltau", "v4rhosigmatau2",
                "v4rholapl3", "v4rholapl2tau", "v4rholapltau2", "v4rhotau3",
                "v4sigma4", "v4sigma3lapl", "v4sigma3tau", "v4sigma2lapl2",
                "v4sigma2lapltau", "v4sigma2tau2", "v4sigmalapl3", "v4sigmalapl2tau",
                "v4sigmalapltau2", "v4sigmatau3", "v4lapl4", "v4lapl3tau",
                "v4lapl2tau2", "v4lapltau3", "v4tau4"
            ]

def _check_arrays(current_arrays, fields, sizes, factor, required):
    """
    A specialized function built to construct and check the sizes of arrays given to the LibXCFunctional class.
    """
    # Nothing supplied so we build it out
    if current_arrays is None:
        current_arrays = {}

    if not required:
        for label in fields:
            current_arrays[label] = None
        return current_arrays

    for label in fields:
        size = sizes[label]
        current_arrays[label] = cupy.empty((factor, size), dtype=np.float64)

    return current_arrays

class _xcfun(ctypes.Structure):
    pass

if _libxc is not None:
    _xc_func_p = ctypes.POINTER(xc_func_type)
    _libxc.xc_func_alloc.restype = _xc_func_p
    _libxc.xc_func_init.argtypes = (_xc_func_p, ctypes.c_int, ctypes.c_int)
    _libxc.xc_func_end.argtypes = (_xc_func_p, )
    _libxc.xc_func_free.argtypes = (_xc_func_p, )

class XCfun:
    def __init__(self, xc, spin):
        self.spin = spin
        self._spin = 1 if spin == 'unpolarized' else 2
        self.xc_func = _libxc.xc_func_alloc()
        if isinstance(xc, str):
            self.func_id = _libxc.xc_functional_get_number(ctypes.c_char_p(xc.encode()))
        else:
            self.func_id = xc
        ret = _libxc.xc_func_init(self.xc_func, self.func_id, self._spin)
        if ret != 0:
            raise RuntimeError('failed to initialize xc fun')
        self._family = dft.libxc.xc_type(xc)

        self.xc_func_sizes = {}
        for attr in dir(self.xc_func.contents.dim):
            if "_" not in attr:
                self.xc_func_sizes[attr] = getattr(self.xc_func.contents.dim, attr)
    def __del__(self):
        if self.xc_func is None:
            return
        if _libxc is not None:
            _libxc.xc_func_end(self.xc_func)
            _libxc.xc_func_free(self.xc_func)

    def needs_laplacian(self):
        return dft.libxc.needs_laplacian(self.func_id)

    rsh_coeff = dft.libxc.rsh_coeff

    def compute(self, inp, output=None, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False):
        # TODO: turn to dft.libxc.eval_xc for do_kxc and do_lxc
        assert not do_kxc
        assert not do_lxc
        if isinstance(inp, cupy.ndarray):
            inp = {"rho": cupy.asarray(inp, dtype=cupy.double)}
        elif isinstance(inp, dict):
            inp = {k: cupy.asarray(v, dtype=cupy.double) for k, v in inp.items()}
        else:
            raise KeyError("Input must have a 'rho' variable or a single array.")

        # How long are we?
        npoints = int(inp["rho"].size / self._spin)
        if (inp["rho"].size % self._spin):
            raise ValueError("Rho input has an invalid shape, must be divisible by %d" % self._spin)

        # Find the right compute function
        args = [self.xc_func, ctypes.c_size_t(npoints)]
        xc_func_sizes = self.xc_func_sizes
        if self._family == 'LDA':
            input_labels   = ["rho"]
            input_num_args = 1
            output_labels = LDA_OUTPUT_LABELS

            # Build input args
            output = _check_arrays(output, output_labels[0:1], xc_func_sizes, npoints, do_exc)
            output = _check_arrays(output, output_labels[1:2], xc_func_sizes, npoints, do_vxc)
            output = _check_arrays(output, output_labels[2:3], xc_func_sizes, npoints, do_fxc)
            output = _check_arrays(output, output_labels[3:4], xc_func_sizes, npoints, do_kxc)
            output = _check_arrays(output, output_labels[4:5], xc_func_sizes, npoints, do_lxc)

            args.extend([   inp[x] for x in  input_labels])
            args.extend([output[x] for x in output_labels])

            out_params = xc_lda_out_params()
            buf_params = xc_lda_out_params()
            buf = copy.deepcopy(output)
            for i, label in enumerate(output_labels):
                if output[label] is not None:
                    setattr(buf_params, label, buf[label].data.ptr)
                    setattr(out_params, label, output[label].data.ptr)
            stream = cupy.cuda.get_current_stream()
            err = libgdft.GDFT_xc_lda(
                stream.ptr,
                self.xc_func,
                npoints,
                inp['rho'].data.ptr,
                ctypes.byref(out_params),
                ctypes.byref(buf_params)
            )
            if err != 0:
                raise RuntimeError('Failed in xc_gga')
        elif self._family == 'GGA':
            input_labels   = ["rho", "sigma"]
            input_num_args = 2
            output_labels = GGA_OUTPUT_LABELS

            # Build input args
            output = _check_arrays(output, output_labels[0:1], xc_func_sizes, npoints, do_exc)
            output = _check_arrays(output, output_labels[1:3], xc_func_sizes, npoints, do_vxc)
            output = _check_arrays(output, output_labels[3:6], xc_func_sizes, npoints, do_fxc)
            output = _check_arrays(output, output_labels[6:10], xc_func_sizes, npoints, do_kxc)
            output = _check_arrays(output, output_labels[10:15], xc_func_sizes, npoints, do_lxc)

            args.extend([   inp[x] for x in  input_labels])
            args.extend([output[x] for x in output_labels])

            out_params = xc_gga_out_params()
            buf_params = xc_gga_out_params()
            buf = copy.deepcopy(output)
            for i, label in enumerate(output_labels):
                if output[label] is not None:
                    setattr(buf_params, label, buf[label].data.ptr)
                    setattr(out_params, label, output[label].data.ptr)

            stream = cupy.cuda.get_current_stream()
            err = libgdft.GDFT_xc_gga(
                stream.ptr,
                self.xc_func,
                npoints,
                inp['rho'].data.ptr,
                inp['sigma'].data.ptr,
                ctypes.byref(out_params),
                ctypes.byref(buf_params)
            )
            if err != 0:
                raise RuntimeError('Failed in xc_gga')

        elif self._family == 'MGGA':
            # Build input args
            if self.needs_laplacian():
                input_labels = ["rho", "sigma", "lapl", "tau"]
            else:
                input_labels = ["rho", "sigma", "tau"]
            input_num_args = 4
            output_labels = MGGA_OUTPUT_LABELS

            # Build input args
            output = _check_arrays(output, output_labels[0:1], xc_func_sizes, npoints, do_exc)
            output = _check_arrays(output, output_labels[1:5], xc_func_sizes, npoints, do_vxc)
            output = _check_arrays(output, output_labels[5:15], xc_func_sizes, npoints, do_fxc)
            output = _check_arrays(output, output_labels[15:35], xc_func_sizes, npoints, do_kxc)
            output = _check_arrays(output, output_labels[35:70], xc_func_sizes, npoints, do_lxc)

            args.extend([   inp[x] for x in  input_labels])
            if not self.needs_laplacian():
                args.insert(-1, cupy.empty((1)))  # Add none ptr to laplacian
            #args.insert(-1, cupy.zeros_like(inp['rho']))
            args.extend([output[x] for x in output_labels])

            out_params = xc_mgga_out_params()
            buf_params = xc_mgga_out_params()
            buf = copy.deepcopy(output)
            for i, label in enumerate(output_labels):
                if output[label] is not None:
                    setattr(buf_params, label, buf[label].data.ptr)
                    setattr(out_params, label, output[label].data.ptr)
            stream = cupy.cuda.get_current_stream()
            lapl = cupy.empty(1)
            err = libgdft.GDFT_xc_mgga(
                stream.ptr,
                self.xc_func,
                npoints,
                inp['rho'].data.ptr,
                inp['sigma'].data.ptr,
                lapl.data.ptr,
                inp['tau'].data.ptr,
                ctypes.byref(out_params),
                ctypes.byref(buf_params)
            )
            if err != 0:
                raise RuntimeError('Failed in xc_mgga')
        else:
            raise KeyError("Functional kind not recognized!")

        return {k: v for k, v in zip(output_labels, args[2+input_num_args:]) if v is not None}

