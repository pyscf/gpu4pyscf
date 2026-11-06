#!/bin/bash
cd /home/abagusetty/gpu4pyscf-testing/sycl_merge/migrate_2026_2
until grep -q "^DONE" summary_s1.txt 2>/dev/null; do sleep 20; done
./run_suite2.sh s3 6
