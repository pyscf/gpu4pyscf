# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import subprocess
import re
import glob

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from distutils.util import get_platform

NAME = 'gpu4pyscf'
AUTHOR = 'PySCF developers'
AUTHOR_EMAIL = None
DESCRIPTION = 'GPU extensions for PySCF'
LICENSE = 'Apache-2.0'
URL = None
DOWNLOAD_URL = None
CLASSIFIERS = None
PLATFORMS = None

# Selects the compute backend for this build. CUDA is the default. To build
# SYCL:
#
#     python setup.py --sycl build
#     CMAKE_CONFIGURE_ARGS=-DUSE_SYCL=ON pip install .
#
# The second form covers pip and other PEP 517 frontends, which do not forward
# unknown flags to setup.py; CMAKE_CONFIGURE_ARGS has to be set for a SYCL
# build regardless, to point cmake at icpx.
def build_backend():
    # --sycl / --cuda are ours, not setuptools', so strip them from argv before
    # setuptools parses it -- an unrecognised global option is a hard error.
    # Precedence: command line, then -DUSE_SYCL=ON in CMAKE_CONFIGURE_ARGS,
    # else CUDA.
    backend = None
    for flag in ('--sycl', '--cuda'):
        while flag in sys.argv:
            sys.argv.remove(flag)
            backend = flag[2:]
    if backend is not None:
        return backend

    if 'USE_SYCL=ON' in os.getenv('CMAKE_CONFIGURE_ARGS', ''):
        return 'sycl'
    return 'cuda'


BACKEND = build_backend()


def get_sycl_version():
    icpx_out = subprocess.check_output(["icpx", "--version"]).decode('utf-8')
    m = re.search(r"[0-9]+\.[0-9]+\.[0-9]+", icpx_out)
    return m.group(0)


def get_cuda_version():
    nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
    m = re.search(r"V[0-9]+.[0-9]+", nvcc_out)
    str_version = m.group(0)[1:]
    major_version, minor_version = str_version.split('.')[:2]
    if major_version == '12' and int(minor_version) < 4:
        # code compiled by 12.4+ may not run on 12.1-12.3
        return major_version + '1'
    return major_version + 'x'

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
        if BACKEND == 'sycl':
            # USE_SYCL defaults OFF in gpu4pyscf/lib/CMakeLists.txt so that a
            # plain CUDA build needs no flags at all. SYCL uses ExchCXX where
            # CUDA uses libxc, and there is no wheel for ExchCXX, so it has to
            # be built from source, hence BUILD_LIBXC=ON rather than the OFF
            # that CUDA passes.
            libxc_arg = '-DBUILD_LIBXC=ON'
            backend_args = ['-DUSE_SYCL=ON']
        else:
            # CUDA takes libxc from the gpu4pyscf-libxc-cuda* wheel listed in
            # install_requires rather than building it (upstream, since #110).
            libxc_arg = '-DBUILD_LIBXC=OFF'
            backend_args = []
        cmd = ['cmake', f'-S{src_dir}', f'-B{dest_dir}', libxc_arg] + backend_args
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
    CUDA_VERSION = '12x'
elif BACKEND == 'sycl':
    package_name = NAME + '-sycl' + get_sycl_version()
else:
    CUDA_VERSION = get_cuda_version()
    package_name = NAME + '-cuda' + CUDA_VERSION

if BACKEND == 'sycl':
    # dpnp replaces cupy, and the SYCL build supplies libxc through ExchCXX.
    INSTALL_REQUIRES = [
        'pyscf>=2.8.0',
        'pyscf-dispersion',
        'dpnp',
        'geometric',
        'packaging',
    ]
else:
    INSTALL_REQUIRES = [
        'pyscf>=2.8.0',
        'pyscf-dispersion',
        # Due to expm in cupyx.scipy.linalg and cutensor 2.0
        f'cupy-cuda{CUDA_VERSION}>=13.0,!=13.4.0',
        'geometric',
        f'gpu4pyscf-libxc-cuda{CUDA_VERSION}==0.8.1',
        'packaging',
    ]

setup(
    name=package_name,
    version=VERSION,
    description=DESCRIPTION,
    license=LICENSE,
    license_files=('LICENSE',),
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
    install_requires=INSTALL_REQUIRES,
)
