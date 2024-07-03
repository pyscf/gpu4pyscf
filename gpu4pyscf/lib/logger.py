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

INFO = lib.logger.INFO
NOTE = lib.logger.NOTE
WARN = lib.logger.WARN
DEBUG = lib.logger.DEBUG
DEBUG1= lib.logger.DEBUG1
DEBUG2= lib.logger.DEBUG2
TIMER_LEVEL = lib.logger.TIMER_LEVEL
flush = lib.logger.flush

if sys.version_info < (3, 0):
    process_clock = time.clock
    perf_counter = time.time
else:
    process_clock = time.process_time
    perf_counter = time.perf_counter


def init_timer(rec):
    if rec.verbose >= TIMER_LEVEL:
        e0 = cupy.cuda.Event()
        e0.record()
        return (process_clock(), perf_counter(), e0)
    elif rec.verbose >= DEBUG:
        return (process_clock(), perf_counter())
    else:
        return process_clock(),

def timer(rec, msg, cpu0=None, wall0=None, gpu0=None):
    if cpu0 is None:
        cpu0 = rec._t0
    if wall0 and gpu0:
        rec._t0, rec._w0, rec._e0 = process_clock(), perf_counter(), cupy.cuda.Event()
        if rec.verbose >= TIMER_LEVEL:
            rec._e0.record()
            rec._e0.synchronize()

            flush(rec, '    CPU time for %50s %9.2f sec, wall time %9.2f sec, GPU time for %9.2f ms'
                  % (msg, rec._t0-cpu0, rec._w0-wall0, cupy.cuda.get_elapsed_time(gpu0,rec._e0)))
        return rec._t0, rec._w0, rec._e0
    elif wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '    CPU time for %50s %9.2f sec, wall time %9.2f sec'
                  % (msg, rec._t0-cpu0, rec._w0-wall0))
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '    CPU time for %50s %9.2f sec' % (msg, rec._t0-cpu0))
        return rec._t0,

def _timer_debug1(rec, msg, cpu0=None, wall0=None, gpu0=None, sync=True):
    if rec.verbose >= DEBUG1:
        return timer(rec, msg, cpu0, wall0, gpu0)
    elif wall0 and gpu0:
        rec._t0, rec._w0, rec._e0 = process_clock(), perf_counter(), cupy.cuda.Event()
        rec._e0.record()
        return rec._t0, rec._w0, rec._e0
    elif wall0:
        rec._t0, rec._w0 = process_clock(), perf_counter()
        return rec._t0, rec._w0
    else:
        rec._t0 = process_clock()
        return rec._t0,

def _timer_debug2(rec, msg, cpu0=None, wall0=None, gpu0=None, sync=True):
    if rec.verbose >= DEBUG2:
        return timer(rec, msg, cpu0, wall0, gpu0)
    return cpu0, wall0, gpu0

info = lib.logger.info
note = lib.logger.note
warn = lib.logger.warn
debug = lib.logger.debug
debug1 = lib.logger.debug1
debug2 = lib.logger.debug2
timer_debug1 = _timer_debug1
timer_debug2 = _timer_debug2

class Logger(lib.logger.Logger):
    def __init__(self, stdout=sys.stdout, verbose=NOTE):
        super().__init__(stdout=stdout, verbose=verbose)
    timer_debug1 = _timer_debug1
    timer_debug2 = _timer_debug2
    timer = timer
    init_timer = init_timer

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
