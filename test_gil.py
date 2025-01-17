import threading
import time

def worker():
    start = time.time()
    for _ in range(10**6):
        time.sleep(1e-3)
        pass  # Simulate some work
    end = time.time()
    print(f"Thread execution time: {end - start:.6f} seconds")

# Run two threads to observe potential GIL contention
threads = [threading.Thread(target=worker) for _ in range(8)]
start_time = time.time()

for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"Total execution time: {time.time() - start_time:.6f} seconds")
