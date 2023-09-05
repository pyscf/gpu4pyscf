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
import shutil

from setuptools import setup, find_packages, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext

NAME = 'gpu4pyscf'
AUTHOR = 'Qiming Sun'
AUTHOR_EMAIL = 'osirpt.sun@gmail.com'
DESCRIPTION = 'GPU extensions for PySCF'
LICENSE = 'GPLv3'
URL = None
DOWNLOAD_URL = None
CLASSIFIERS = None
PLATFORMS = None


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


class CMakeBuildExt(build_ext):
    def run(self):
        extension = self.extensions[0]
        assert extension.name == 'gpu4pyscf_lib_placeholder'
        src_path = os.path.abspath(os.path.join(__file__, '..', 'gpu4pyscf', 'lib'))
        dest_path = os.path.join(self.build_temp, 'gpu4pyscf')
        os.makedirs(dest_path, exist_ok=True)
        self.build_cmake(src_path, dest_path)

    def build_cmake(self, src_dir, dest_dir, *,
                    configure_args=None, build_args=None):
        self.announce('Configuring extensions', level=3)
        cmd = ['cmake', f'-S{src_dir}', f'-B{dest_dir}']
        if configure_args is None:
            configure_args = os.getenv('CMAKE_CONFIGURE_ARGS')

        if configure_args:
            cmd.extend(configure_args.split(' '))

        self.spawn(cmd)

        self.announce('Building binaries', level=3)
        cmd = ['cmake', '--build', dest_dir, '-j8']
        if build_args is None:
            build_args = os.getenv('CMAKE_BUILD_ARGS')

        if build_args:
            cmd.extend(build_args.split(' '))

        if self.dry_run:
            self.announce(' '.join(cmd))
        else:
            self.spawn(cmd)

    # To remove the infix string like cpython-37m-x86_64-linux-gnu.so
    # Python ABI updates since 3.5
    # https://www.python.org/dev/peps/pep-3149/
    def get_ext_filename(self, ext_name):
        ext_path = ext_name.split('.')
        filename = build_ext.get_ext_filename(self, ext_name)
        name, ext_suffix = os.path.splitext(filename)
        return os.path.join(*ext_path) + ext_suffix


# Here to change the order of sub_commands to ['build_py', ..., 'build_ext']
# C extensions by build_ext are installed in source directory.
# build_py then copy all .so files into "build_ext.build_lib" directory.
# We have to ensure build_ext being executed earlier than build_py.
# A temporary workaround is to modifying the order of sub_commands in build class
from distutils.command.build import build

build.sub_commands = ([c for c in build.sub_commands if c[0] == 'build_ext'] +
                      [c for c in build.sub_commands if c[0] != 'build_ext'])

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    package_dir={'gpu4pyscf': 'gpu4pyscf'},  # packages are under directory pyscf
    # include *.so *.dat files. They are now placed in MANIFEST.in
    
    include_package_data=True,  # include everything in source control
    packages=[*find_namespace_packages('.'), 'gpu4pyscf', 'gpu4pyscf.lib'],
    tests_require=[
        "pytest==7.2.0",
        "pytest-cov==4.0.0",
        "pytest-cover==3.0.0",
        "pytest-coverage==0.0",
    ],
    # The ext_modules placeholder is to ensure build_ext getting initialized
    ext_modules=[Extension('gpu4pyscf_lib_placeholder', [])],
    cmdclass={'build_ext': CMakeBuildExt},
    install_requires=[
        'pyscf>=2.3.0',
        'cupy-cuda11x>=12.0',
        'dftd3==0.7.0',
        'dftd4==3.5.0',
        'geometric'
    ],
)
