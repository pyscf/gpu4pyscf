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

import sys
import time
import cupy
from pyscf import lib

from pyscf.lib import parameters as param
import pyscf.__config__

NOTE = lib.logger.NOTE
WARN = lib.logger.WARN
DEBUG = lib.logger.DEBUG
DEBUG1= lib.logger.DEBUG1

if sys.version_info < (3, 0):
    process_clock = time.clock
    perf_counter = time.time
else:
    process_clock = time.process_time
    perf_counter = time.perf_counter

def _timer_debug1(rec, msg, cpu0=None, wall0=None, sync=True):
    if rec.verbose >= DEBUG1:
        if(sync): cupy.cuda.stream.get_current_stream().synchronize()
        return timer(rec, msg, cpu0, wall0)
    elif wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        return rec._t0

info = lib.logger.info
debug = lib.logger.debug
debug1 = lib.logger.debug1
timer = lib.logger.timer
timer_debug1 = _timer_debug1

class Logger(lib.logger.Logger):
    def __init__(self, stdout=sys.stdout, verbose=NOTE):
        super().__init__(stdout=stdout, verbose=verbose)
    timer_debug1 = _timer_debug1

def new_logger(rec=None, verbose=None):
    '''Create and return a :class:`Logger` object

    Args:
        rec : An object which carries the attributes stdout and verbose

        verbose : a Logger object, or integer or None
            The verbose level. If verbose is a Logger object, the Logger
            object is returned. If verbose is not specified (None),
            rec.verbose will be used in the new Logger object.
    '''
    if isinstance(verbose, Logger):
        log = verbose
    elif isinstance(verbose, int):
        if getattr(rec, 'stdout', None):
            log = Logger(rec.stdout, verbose)
        else:
            log = Logger(sys.stdout, verbose)
    else:
        log = Logger(rec.stdout, rec.verbose)
    return log

