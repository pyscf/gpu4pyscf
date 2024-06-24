#!/usr/bin/env python

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
import sys
import subprocess
import re
import glob

from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from distutils.util import get_platform

NAME = 'gpu4pyscf'
AUTHOR = 'Qiming Sun'
AUTHOR_EMAIL = 'osirpt.sun@gmail.com'
DESCRIPTION = 'GPU extensions for PySCF'
LICENSE = 'GPLv3'
URL = None
DOWNLOAD_URL = None
CLASSIFIERS = None
PLATFORMS = None

def get_cuda_version():
    nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
    m = re.search(r"V[0-9]+.[0-9]+", nvcc_out)
    str_version = m.group(0)[1:]
    return str_version[:2]+'x'

def get_version():
    topdir = os.path.abspath(os.path.join(__file__, '..'))
    module_path = os.path.join(topdir, 'gpu4pyscf')
    for version_file in ['__init__.py', '_version.py']:
        version_file = os.path.join(module_path, version_file)
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                for line in f.readlines():
                    if line.startswith('__version__'):
                        delim = '"' if '"' in line else "'"
                        return line.split(delim)[1]
    raise ValueError("Version string not found")


VERSION = get_version()


class CMakeBuildPy(build_py):
    def run(self):
        self.plat_name = get_platform()
        self.build_base = 'build'
        self.build_lib = os.path.join(self.build_base, 'lib')
        self.build_temp = os.path.join(self.build_base, f'temp.{self.plat_name}')

        self.announce('Configuring extensions', level=3)
        src_dir = os.path.abspath(os.path.join(__file__, '..', 'gpu4pyscf', 'lib'))
        dest_dir = os.path.join(self.build_temp, 'gpu4pyscf')
        cmd = ['cmake', f'-S{src_dir}', f'-B{dest_dir}', '-DBUILD_LIBXC=OFF']
        configure_args = os.getenv('CMAKE_CONFIGURE_ARGS')
        if configure_args:
            cmd.extend(configure_args.split(' '))
        self.spawn(cmd)

        self.announce('Building binaries', level=3)
        cmd = ['cmake', '--build', dest_dir, '-j', '8']
        build_args = os.getenv('CMAKE_BUILD_ARGS')
        if build_args:
            cmd.extend(build_args.split(' '))
        if self.dry_run:
            self.announce(' '.join(cmd))
        else:
            self.spawn(cmd)

        super().run()

# build_py will produce plat_name = 'any'. Patch the bdist_wheel to change the
# platform tag because the C extensions are platform dependent.
# For setuptools<70
from wheel.bdist_wheel import bdist_wheel
initialize_options_1 = bdist_wheel.initialize_options
def initialize_with_default_plat_name(self):
    initialize_options_1(self)
    self.plat_name = get_platform()
    self.plat_name_supplied = True
bdist_wheel.initialize_options = initialize_with_default_plat_name

# For setuptools>=70
try:
    from setuptools.command.bdist_wheel import bdist_wheel
    initialize_options_2 = bdist_wheel.initialize_options
    def initialize_with_default_plat_name(self):
        initialize_options_2(self)
        self.plat_name = get_platform()
        self.plat_name_supplied = True
    bdist_wheel.initialize_options = initialize_with_default_plat_name
except ImportError:
    pass

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
    package_dir={'gpu4pyscf': 'gpu4pyscf'},  # packages are under directory pyscf
    # include *.so *.dat files. They are now placed in MANIFEST.in
    include_package_data=True,  # include everything in source control
    packages=find_packages(exclude=['*test*', '*examples*', '*docker*']),
    tests_require=[
        "pytest==7.2.0",
        "pytest-cov==4.0.0",
        "pytest-cover==3.0.0",
        "pytest-coverage==0.0",
    ],
    cmdclass={'build_py': CMakeBuildPy},
    install_requires=[
        'pyscf~=2.6.0',
        'pyscf-dispersion',
        f'cupy-cuda{CUDA_VERSION}',
        'geometric',
        f'gpu4pyscf-libxc-cuda{CUDA_VERSION}',
    ]
)
