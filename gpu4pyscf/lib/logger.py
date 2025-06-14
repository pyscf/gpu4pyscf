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

import sys
import time
import cupy
from pyscf import lib

INFO = lib.logger.INFO
NOTE = lib.logger.NOTE
WARN = lib.logger.WARN
DEBUG = lib.logger.DEBUG
DEBUG1= lib.logger.DEBUG1
DEBUG2= lib.logger.DEBUG2
TIMER_LEVEL = lib.logger.TIMER_LEVEL
flush = lib.logger.flush

process_clock = time.process_time
perf_counter = time.perf_counter


def init_timer(rec):
    wall = e0 = None
    if rec.verbose >= TIMER_LEVEL:
        e0 = cupy.cuda.Event()
        e0.record()
        wall = perf_counter()
    return (process_clock(), wall, e0)

def timer(rec, msg, cpu0=None, wall0=None, gpu0=None):
    if gpu0:
        t0, w0, e0 = process_clock(), perf_counter(), cupy.cuda.Event()
        e0.record()
        if rec.verbose >= TIMER_LEVEL:
            e0.synchronize()
            flush(rec, '    CPU time for %-50s %9.2f sec, wall time %9.2f sec, GPU time %9.2f ms'
                  % (msg, t0-cpu0, w0-wall0, cupy.cuda.get_elapsed_time(gpu0,e0)))
        return t0, w0, e0
    elif wall0:
        t0, w0 = process_clock(), perf_counter()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '    CPU time for %s %9.2f sec, wall time %9.2f sec'
                  % (msg, t0-cpu0, w0-wall0))
        return t0, w0
    else:
        t0 = process_clock()
        if rec.verbose >= TIMER_LEVEL:
            flush(rec, '    CPU time for %s %9.2f sec' % (msg, t0-cpu0))
        return t0,

def timer_silent(rec, cpu0=None, wall0=None, gpu0=None):
    if gpu0:
        t0, w0, e0 = process_clock(), perf_counter(), cupy.cuda.Event()
        e0.record()
        e0.synchronize()
        return t0-cpu0, w0-wall0, cupy.cuda.get_elapsed_time(gpu0,e0)
    elif wall0:
        t0, w0 = process_clock(), perf_counter()
        return t0-cpu0, w0-wall0
    else:
        t0 = process_clock()
        return t0-cpu0,


def _timer_debug1(rec, msg, cpu0=None, wall0=None, gpu0=None, sync=True):
    if rec.verbose >= DEBUG1:
        return timer(rec, msg, cpu0, wall0, gpu0)
    elif gpu0:
        t0, w0, e0 = process_clock(), perf_counter(), cupy.cuda.Event()
        e0.record()
        return t0, w0, e0
    elif wall0:
        t0, w0 = process_clock(), perf_counter()
        return t0, w0
    else:
        t0 = process_clock()
        return t0,

def _timer_debug2(rec, msg, cpu0=None, wall0=None, gpu0=None, sync=True):
    if rec.verbose >= DEBUG2:
        return timer(rec, msg, cpu0, wall0, gpu0)
    elif gpu0:
        t0, w0, e0 = process_clock(), perf_counter(), cupy.cuda.Event()
        e0.record()
        return t0, w0, e0
    elif wall0:
        t0, w0 = process_clock(), perf_counter()
        return t0, w0
    else:
        t0 = process_clock()
        return t0,

def print_mem_info(rec):
    mempool = cupy.get_default_memory_pool()
    used_mem = mempool.used_bytes()
    free_mem = mempool.free_bytes()
    free_blocks = mempool.n_free_blocks()
    mem_avail = cupy.cuda.runtime.memGetInfo()[0]
    flush(rec, f'mem_info: unallocated={mem_avail/1024**2:.2f} MB, '
          f'used={used_mem/1024**2:.2f} MB, free={free_mem/1024**2:.2f} MB, '
          f'free_blocks={free_blocks}')
    return mem_avail + free_mem

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
    timer_silent = timer_silent
    print_mem_info = print_mem_info

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
