#!/usr/bin/env python

# gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
#
# Copyright (C) 2023 Qiming Sun
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
import sys
import subprocess
import re

from setuptools import setup, find_packages, Extension, find_namespace_packages
from setuptools.command.build_py import build_py
from distutils.util import get_platform

NAME = 'gpu4pyscf-libxc'
AUTHOR = 'Qiming Sun'
AUTHOR_EMAIL = 'osirpt.sun@gmail.com'
DESCRIPTION = 'GPU extensions for PySCF'
LICENSE = 'GPLv3'
URL = None
DOWNLOAD_URL = None
CLASSIFIERS = None
PLATFORMS = None
VERSION = '0.1'

def get_cuda_version():
    nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
    m = re.search(r"V[0-9]+.[0-9]+", nvcc_out)
    str_version = m.group(0)[1:]
    return str_version[:2]+'x'


# build_py will produce plat_name = 'any'. Patch the bdist_wheel to change the
# platform tag because the C extensions are platform dependent.
from wheel.bdist_wheel import bdist_wheel
initialize_options = bdist_wheel.initialize_options
def initialize_with_default_plat_name(self):
    initialize_options(self)
    self.plat_name = get_platform()
bdist_wheel.initialize_options = initialize_with_default_plat_name

if 'sdist' in sys.argv:
    # The sdist release
    package_name = NAME
    CUDA_VERSION = '11x'
else:
    CUDA_VERSION = get_cuda_version()
    package_name = NAME + '-cuda' + CUDA_VERSION

setup(
    name=package_name,
    version=VERSION,
    description=DESCRIPTION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    package_data={'gpu4pyscf.lib.deps.lib': ['libxc.so']},
    packages=['gpu4pyscf.lib.deps.lib'],
)
