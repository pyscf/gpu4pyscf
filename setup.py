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
import shutil

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

def get_cuda_version():
    nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
    m = re.search(r"V[0-9]+.[0-9]+", nvcc_out)
    str_version = m.group(0)[1:]
    major_version, minor_version = str_version.split('.')[:2]
    if major_version == '12' and int(minor_version) < 4:
        # code compiled by 12.4+ may not run on 12.1-12.3
        return major_version + '1'
    return major_version + 'x'

# ---------------------------------------------------------------------------
# GPU backend selection: NVIDIA CUDA (default) or AMD HIP/ROCm.
# The CUDA path is unchanged; the HIP path is only taken when explicitly
# requested (USE_HIP=1) or auto-detected on a ROCm-only machine (no nvcc).
# ---------------------------------------------------------------------------
def _env_flag(name):
    """True/False for an explicitly set env var, or None if it is unset."""
    val = os.getenv(name)
    if val is None:
        return None
    return val.strip().lower() in ('1', 'on', 'true', 'yes')

def find_rocm_path():
    """Best-effort location of a ROCm/HIP install (env first, then hipconfig)."""
    for var in ('ROCM_PATH', 'ROCM_HOME', 'HIP_PATH'):
        path = os.getenv(var)
        if path and os.path.isdir(path):
            return path
    hipconfig = shutil.which('hipconfig')
    if hipconfig:
        try:
            return subprocess.check_output([hipconfig, '-p']).decode('utf-8').strip()
        except Exception:
            pass
    return None

def use_hip():
    """Select the AMD HIP/ROCm backend instead of NVIDIA CUDA.

    An explicit ``USE_HIP`` environment variable (1/0) always wins. Otherwise
    the backend is auto-detected: CUDA when ``nvcc`` is on PATH, else HIP when a
    ROCm install is found. On machines with a CUDA toolkit the default (CUDA) is
    unchanged.
    """
    flag = _env_flag('USE_HIP')
    if flag is not None:
        return flag
    if shutil.which('nvcc'):
        return False
    return find_rocm_path() is not None

def get_rocm_version():
    """ROCm major-version tag for the package name, e.g. '7x'. Best-effort."""
    out = ''
    hipconfig = shutil.which('hipconfig')
    if hipconfig:
        try:
            out = subprocess.check_output([hipconfig, '--version']).decode('utf-8')
        except Exception:
            out = ''
    m = re.search(r'(\d+)\.', out) or re.search(r'(\d+)\.\d+', find_rocm_path() or '')
    return (m.group(1) if m else 'x') + 'x'

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

USE_HIP = use_hip()


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
        if USE_HIP:
            cmd.extend(self._setup_hip_build())
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

    def _setup_hip_build(self):
        """Environment and CMake flags for an AMD HIP/ROCm build.

        Mirrors rocm/build_gpu4pyscf.sh so ``pip install`` reproduces the
        script's build: export the ROCm paths and select the HIP language via
        -DUSE_HIP=ON. The GPU architecture is auto-detected by CMake
        (rocm_agent_enumerator); override with GPU_ARCHITECTURES=gfxXXXX.
        """
        rocm_path = find_rocm_path()
        if not rocm_path:
            raise RuntimeError(
                "USE_HIP is set but no ROCm installation was found. "
                "Set ROCM_PATH (or put hipconfig on PATH).")
        os.environ.setdefault('ROCM_PATH', rocm_path)
        os.environ.setdefault('ROCM_HOME', rocm_path)
        os.environ.setdefault('HIP_PATH', rocm_path)
        os.environ.setdefault('HIP_PLATFORM', 'amd')
        bindir = os.path.join(rocm_path, 'bin')
        os.environ['PATH'] = bindir + os.pathsep + os.environ.get('PATH', '')
        libdirs = os.pathsep.join(os.path.join(rocm_path, d) for d in ('lib', 'lib64'))
        os.environ['LD_LIBRARY_PATH'] = libdirs + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')

        args = ['-DUSE_HIP=ON']
        hip_compiler = os.getenv('CMAKE_HIP_COMPILER',
                                 os.path.join(rocm_path, 'llvm', 'bin', 'clang++'))
        if os.path.exists(hip_compiler):
            args.append(f'-DCMAKE_HIP_COMPILER={hip_compiler}')
        gpu_arch = os.getenv('GPU_ARCHITECTURES')
        if gpu_arch:
            args.append(f'-DGPU_ARCHITECTURES={gpu_arch}')
        return args

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

if USE_HIP and 'sdist' not in sys.argv:
    # AMD ROCm/HIP build. CuPy for ROCm and (optionally) PyTorch-ROCm are
    # provided by the environment -- there is no cupy-rocm / gpu4pyscf-libxc-rocm
    # wheel to depend on -- so only the backend-agnostic requirements are declared.
    package_name = NAME + '-rocm' + get_rocm_version()
    install_requires = [
        'pyscf>=2.8.0',
        'geometric',
        'packaging',
    ]
else:
    # NVIDIA CUDA build, and the backend-agnostic sdist source release.
    package_name = NAME
    if 'sdist' in sys.argv:
        CUDA_VERSION = '12x'
    else:
        CUDA_VERSION = get_cuda_version()
        if os.getenv('GPU4PYSCF_BUILD_CUDA_WHEEL'):
            package_name = NAME + '-cuda' + CUDA_VERSION
    install_requires = [
        'pyscf>=2.8.0',
        f'cupy-cuda{CUDA_VERSION}>=13.0,!=13.4.0', # Due to expm in cupyx.scipy.linalg and cutensor 2.0
        'geometric',
        f'gpu4pyscf-libxc-cuda{CUDA_VERSION}>=0.8.1',
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
    extras_require={'dispersion': ['pyscf-dispersion>=1.7.0']},
    install_requires=install_requires,
)
