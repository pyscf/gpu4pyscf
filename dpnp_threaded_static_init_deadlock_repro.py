"""
dpnp: deadlock between the GIL and a C++ function-local static guard in
dpnp/tensor/_tensor_elementwise_impl

Two threads doing ordinary elementwise dpnp arithmetic on a shared queue can
wedge the whole process on the very first use of an elementwise kernel:

  Thread A (holds the GIL):
      slot_nb_multiply -> ... -> _tensor_elementwise_impl.so
                       -> __cxa_guard_acquire        <-- blocked
  Threads B..N:
      take_gil                                        <-- blocked

A function-local `static` inside the elementwise implementation is being
initialised by one thread while it holds the GIL; the guard makes every other
thread wait for that initialisation, but the initialising thread cannot make
progress because the other threads hold resources it needs, and they cannot run
because they are queued on the GIL. Classic lock-order inversion between the
GIL and the C++ static-init guard.

Verified with gdb on a hung run (Intel Data Center GPU Max 1550, Level Zero):

    Thread 5  #1  __cxa_guard_acquire (g=0x...)
              #2-#5  dpnp/tensor/_tensor_elementwise_impl.cpython-312.so
              #16 slot_nb_multiply                    [holds GIL]
    Thread 3  take_gil
    Thread 4  take_gil
    Thread 12 take_gil
    Thread 1  main, blocked in Thread.join()

No `sycl::event` wait, no host task and no finalizer is involved -- this is
distinct from intel/llvm#22943.

Run with a timeout; on failure it hangs forever and prints nothing:

    timeout 120 python dpnp_threaded_static_init_deadlock_repro.py

Expected on success: 40 lines of "round N/40 ok" then "completed".
Observed: hangs before printing "round 1/40 ok".
"""
import threading

import dpctl
import dpnp

NTHREADS = 8
NROUNDS = 40
N = 1 << 14

print("dpctl", dpctl.__version__, "| dpnp", dpnp.__version__, flush=True)

q = dpctl.SyclQueue(dpctl.SyclDevice("gpu"), property="in_order")
print("device:", q.sycl_device.name, "| in_order:", q.is_in_order, flush=True)


def worker():
    a = dpnp.ones(N, sycl_queue=q)
    b = dpnp.arange(N, dtype="f8", sycl_queue=q)
    for _ in range(6):
        b = b * 1.000001 + a      # <-- elementwise kernel, static-init guard
    float(b[0])


for r in range(NROUNDS):
    ts = [threading.Thread(target=worker) for _ in range(NTHREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"round {r + 1}/{NROUNDS} ok", flush=True)

print("completed without deadlock", flush=True)
