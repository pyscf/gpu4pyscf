import dpctl, dpnp, time
q = dpctl.SyclQueue("level_zero:gpu", property="in_order")
# idle queue: barrier event wait, repeated, back-to-back (the 2026.1 abort pattern)
evs = [q.submit_barrier() for _ in range(5)]
for i, e in enumerate(evs):
    e.wait(); print("idle b%d wait ok status=%s" % (i, e.execution_status))
# busy queue
a = dpnp.ones(1 << 26, dtype="f8", sycl_queue=q)
for _ in range(20): a = a * 1.0000001 + 1e-9
t = time.time(); e = q.submit_barrier(); e.wait(); print("busy wait %.3fs ok" % (time.time() - t))
for i in range(200):
    e1 = q.submit_barrier(); e2 = q.submit_barrier(); e1.wait(); e2.wait(); e1.wait()
print("stress ok")
