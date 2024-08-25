# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
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

"""
Wrappers to the LibXC C structs
"""

"""
Modified source code from libXC
"""

import ctypes
import numpy as np

XC_FAMILY_LDA     =      1
XC_FAMILY_GGA     =      2
XC_FAMILY_MGGA    =      4
XC_FAMILY_LCA     =      8
XC_FAMILY_OEP     =     16
XC_FAMILY_HYB_GGA =     32
XC_FAMILY_HYB_MGGA=     64
XC_FAMILY_HYB_LDA =    128

class func_reference_type(ctypes.Structure):
    """
    Holds reference data for the LibXC functional
    """
    _fields_ = [("ref", ctypes.c_char_p), ("doi", ctypes.c_char_p),
                ("bibtex", ctypes.c_char_p), ("key", ctypes.c_char_p)]


class func_params_type(ctypes.Structure):
    """
    Holds user defined parameters and their description.
    """
    _fields_ = [("value", ctypes.c_double), ("description", ctypes.c_char_p)]


class xc_func_info_type(ctypes.Structure):
    """
    Holds LibXC information about a single functional primitive.
    """
    _fields_ = [
        ("number", ctypes.c_int),
        ("kind", ctypes.c_int),
        ("name", ctypes.c_char_p),
        ("family", ctypes.c_int),
        ("refs", ctypes.POINTER(func_reference_type)),
        ("flags", ctypes.c_int),
        ("dens_threshold", ctypes.c_double),
        ("n_ext_params", ctypes.c_int),
        ("ext_params", ctypes.POINTER(func_params_type)),
        ("set_ext_params", ctypes.c_void_p),
        ("init", ctypes.c_void_p),
        ("end", ctypes.c_void_p),
        ("lda", ctypes.c_void_p),
        ("gga", ctypes.c_void_p),
        ("mgga", ctypes.c_void_p),
    ]


class xc_dimensions(ctypes.Structure):
    """
    Holds dimensions of the several arrays.
    """
    _fields_ = [
         ("rho", ctypes.c_int),
         ("sigma", ctypes.c_int),
         ("lapl", ctypes.c_int),
         ("tau", ctypes.c_int),

         ("zk", ctypes.c_int),

         ("vrho", ctypes.c_int),
         ("vsigma", ctypes.c_int),
         ("vlapl", ctypes.c_int),
         ("vtau", ctypes.c_int), 

         ("v2rho2", ctypes.c_int),
         ("v2rhosigma", ctypes.c_int),
         ("v2rholapl", ctypes.c_int),
         ("v2rhotau", ctypes.c_int),
         ("v2sigma2", ctypes.c_int),
         ("v2sigmalapl", ctypes.c_int),
         ("v2sigmatau", ctypes.c_int),
         ("v2lapl2", ctypes.c_int),
         ("v2lapltau", ctypes.c_int),
         ("v2tau2", ctypes.c_int),

         ("v3rho3", ctypes.c_int),
         ("v3rho2sigma", ctypes.c_int),
         ("v3rho2lapl", ctypes.c_int),
         ("v3rho2tau", ctypes.c_int),
         ("v3rhosigma2", ctypes.c_int),
         ("v3rhosigmalapl", ctypes.c_int),
         ("v3rhosigmatau", ctypes.c_int),
         ("v3rholapl2", ctypes.c_int),
         ("v3rholapltau", ctypes.c_int),
         ("v3rhotau2", ctypes.c_int),
         ("v3sigma3", ctypes.c_int),
         ("v3sigma2lapl", ctypes.c_int),
         ("v3sigma2tau", ctypes.c_int),
         ("v3sigmalapl2", ctypes.c_int),
         ("v3sigmalapltau", ctypes.c_int),
         ("v3sigmatau2", ctypes.c_int),
         ("v3lapl3", ctypes.c_int),
         ("v3lapl2tau", ctypes.c_int),
         ("v3lapltau2", ctypes.c_int),
         ("v3tau3", ctypes.c_int),

         ("v4rho4", ctypes.c_int),
         ("v4rho3sigma", ctypes.c_int),
         ("v4rho3lapl", ctypes.c_int),
         ("v4rho3tau", ctypes.c_int),
         ("v4rho2sigma2", ctypes.c_int),
         ("v4rho2sigmalapl", ctypes.c_int),
         ("v4rho2sigmatau", ctypes.c_int),
         ("v4rho2lapl2", ctypes.c_int),
         ("v4rho2lapltau", ctypes.c_int),
         ("v4rho2tau2", ctypes.c_int),
         ("v4rhosigma3", ctypes.c_int),
         ("v4rhosigma2lapl", ctypes.c_int),
         ("v4rhosigma2tau", ctypes.c_int),
         ("v4rhosigmalapl2", ctypes.c_int),
         ("v4rhosigmalapltau", ctypes.c_int),
         ("v4rhosigmatau2", ctypes.c_int),
         ("v4rholapl3", ctypes.c_int),
         ("v4rholapl2tau", ctypes.c_int),
         ("v4rholapltau2", ctypes.c_int),
         ("v4rhotau3", ctypes.c_int),
         ("v4sigma4", ctypes.c_int),
         ("v4sigma3lapl", ctypes.c_int),
         ("v4sigma3tau", ctypes.c_int),
         ("v4sigma2lapl2", ctypes.c_int),
         ("v4sigma2lapltau", ctypes.c_int),
         ("v4sigma2tau2", ctypes.c_int),
         ("v4sigmalapl3", ctypes.c_int),
         ("v4sigmalapl2tau", ctypes.c_int),
         ("v4sigmalapltau2", ctypes.c_int),
         ("v4sigmatau3", ctypes.c_int),
         ("v4lapl4", ctypes.c_int),
         ("v4lapl3tau", ctypes.c_int),
         ("v4lapl2tau2", ctypes.c_int),
         ("v4lapltau3", ctypes.c_int),
         ("v4tau4", ctypes.c_int)]


class xc_func_type(ctypes.Structure):
    """
    The primary xc_func_type used to hold all data pertaining to a given
    LibXC functional
    """
    _fields_ = [
        ("info", ctypes.POINTER(xc_func_info_type)),  # const xc_func_info_type *info;
        ("nspin", ctypes.c_int),
        ("n_func_aux", ctypes.c_int),
        ("xc_func_type", ctypes.c_void_p),
        ("mix_coef", ctypes.POINTER(ctypes.c_double)),

        # Hybrids
        ("cam_omega", ctypes.c_double),
        ("cam_alpha", ctypes.c_double),
        ("cam_beta", ctypes.c_double),

        # VV10
        ("nlc_b", ctypes.c_double),
        ("nlc_C", ctypes.c_double),

        ("dim", xc_dimensions),

        # parameters
        ("ext_params", ctypes.POINTER(ctypes.c_double)),
        ("params", ctypes.c_void_p),  # void *params;
        
        ("dens_threshold", ctypes.c_double),
        ("zeta_threshold", ctypes.c_double),
        ("sigma_threshold", ctypes.c_double),
        ("tau_threshold", ctypes.c_double)
    ]

class xc_lda_out_params(ctypes.Structure):
    """
    Holds the output parameters for LDA functions
    """
    _fields_ = [
        ("zk", ctypes.c_void_p),
        ("vrho", ctypes.c_void_p),
        ("v2rho2", ctypes.c_void_p),
        ("v3rho3", ctypes.c_void_p),
        ("v4rho4", ctypes.c_void_p),
    ]

class xc_gga_out_params(ctypes.Structure):
    """
    Holds the output parameters for GGA functions
    """
    _fields_ = [
        ("zk", ctypes.c_void_p),
        ("vrho", ctypes.c_void_p),
        ("vsigma", ctypes.c_void_p),
        ("v2rho2", ctypes.c_void_p),
        ("v2rhosigma", ctypes.c_void_p),
        ("v2sigma2", ctypes.c_void_p),
        ("v3rho3", ctypes.c_void_p),
        ("v3rho2sigma", ctypes.c_void_p),
        ("v3rhosigma2", ctypes.c_void_p),
        ("v3sigma3", ctypes.c_void_p),
        ("v4rho4", ctypes.c_void_p),
        ("v4rho3sigma", ctypes.c_void_p),
        ("v4rho2sigma2", ctypes.c_void_p),
        ("v4rhosigma3", ctypes.c_void_p),
        ("v4sigma4", ctypes.c_void_p),
    ]

class xc_mgga_out_params(ctypes.Structure):
    """
    Holds the output parameters for MGGA functions
    """
    _fields_ = [
        ("zk", ctypes.c_void_p),

        ("vrho", ctypes.c_void_p),
        ("vsigma", ctypes.c_void_p),
        ("vlapl", ctypes.c_void_p),
        ("vtau", ctypes.c_void_p),

        ("v2rho2", ctypes.c_void_p),
        ("v2rhosigma", ctypes.c_void_p),
        ("v2rholapl", ctypes.c_void_p),
        ("v2rhotau", ctypes.c_void_p),
        ("v2sigma2", ctypes.c_void_p),
        ("v2sigmalapl", ctypes.c_void_p),
        ("v2sigmatau", ctypes.c_void_p),
        ("v2lapl2", ctypes.c_void_p),
        ("v2lapltau", ctypes.c_void_p),
        ("v2tau2", ctypes.c_void_p),

        ("v3rho3", ctypes.c_void_p),
        ("v3rho2sigma", ctypes.c_void_p),
        ("v3rho2lapl", ctypes.c_void_p),
        ("v3rho2tau", ctypes.c_void_p),
        ("v3rhosigma2", ctypes.c_void_p),
        ("v3rhosigmalapl", ctypes.c_void_p),
        ("v3rhosigmatau", ctypes.c_void_p),
        ("v3rholapl2", ctypes.c_void_p),
        ("v3rholapltau", ctypes.c_void_p),
        ("v3rhotau2", ctypes.c_void_p),
        ("v3sigma3", ctypes.c_void_p),
        ("v3sigma2lapl", ctypes.c_void_p),
        ("v3sigma2tau", ctypes.c_void_p),
        ("v3sigmalapl2", ctypes.c_void_p),
        ("v3sigmalapltau", ctypes.c_void_p),
        ("v3sigmatau2", ctypes.c_void_p),
        ("v3lapl3", ctypes.c_void_p),
        ("v3lapl2tau", ctypes.c_void_p),
        ("v3lapltau2", ctypes.c_void_p),
        ("v3tau3", ctypes.c_void_p),

        ("v4rho4", ctypes.c_void_p),
        ("v4rho3sigma", ctypes.c_void_p),
        ("v4rho3lapl", ctypes.c_void_p),
        ("v4rho3tau", ctypes.c_void_p),
        ("v4rho2sigma2", ctypes.c_void_p),
        ("v4rho2sigmalapl", ctypes.c_void_p),
        ("v4rho2sigmatau", ctypes.c_void_p),
        ("v4rho2lapl2", ctypes.c_void_p),
        ("v4rho2lapltau", ctypes.c_void_p),
        ("v4rho2tau2", ctypes.c_void_p),
        ("v4rhosigma3", ctypes.c_void_p),
        ("v4rhosigma2lapl", ctypes.c_void_p),
        ("v4rhosigma2tau", ctypes.c_void_p),
        ("v4rhosigmalapl2", ctypes.c_void_p),
        ("v4rhosigmalapltau", ctypes.c_void_p),
        ("v4rhosigmatau2", ctypes.c_void_p),
        ("v4rholapl3", ctypes.c_void_p),
        ("v4rholapl2tau", ctypes.c_void_p),
        ("v4rholapltau2", ctypes.c_void_p),
        ("v4rhotau3", ctypes.c_void_p),
        ("v4sigma4", ctypes.c_void_p),
        ("v4sigma3lapl", ctypes.c_void_p),
        ("v4sigma3tau", ctypes.c_void_p),
        ("v4sigma2lapl2", ctypes.c_void_p),
        ("v4sigma2lapltau", ctypes.c_void_p),
        ("v4sigma2tau2", ctypes.c_void_p),
        ("v4sigmalapl3", ctypes.c_void_p),
        ("v4sigmalapl2tau", ctypes.c_void_p),
        ("v4sigmalapltau2", ctypes.c_void_p),
        ("v4sigmatau3", ctypes.c_void_p),
        ("v4lapl4", ctypes.c_void_p),
        ("v4lapl3tau", ctypes.c_void_p),
        ("v4lapl2tau2", ctypes.c_void_p),
        ("v4lapltau3", ctypes.c_void_p),
        ("v4tau4", ctypes.c_void_p),
    ]
