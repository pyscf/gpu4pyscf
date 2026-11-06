import dpctl, dpnp, time
q = dpctl.SyclQueue("level_zero:gpu", property="in_order")
a = dpnp.ones(1 << 28, dtype="f8", sycl_queue=q)   # 2 GB
b = dpnp.ones(1 << 28, dtype="f8", sycl_queue=q)
for trial in range(3):
    t0 = time.time()
    for _ in range(60): b = a * b + 1e-9
    t_sub = time.time() - t0
    e = q.submit_barrier()
    t1 = time.time(); e.wait(); t_ev = time.time() - t1
    t2 = time.time(); q.wait(); t_q = time.time() - t2
    print("trial %d: submit %.2fs, barrier wait %.2fs, then queue.wait %.3fs" % (trial, t_sub, t_ev, t_q))
