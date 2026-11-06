#!/bin/bash
# One full pass on oneAPI 2026.2.0. One pytest per file, 1500 s timeout, 8 pinned cores per tile
# (socket 1, tiles 6-11; cores 52/103 reserved). Usage: run_suite.sh <pass-name>
R=/home/abagusetty/gpu4pyscf-testing/sycl_merge
cd $R
. $R/migrate_2026_2/env.sh
PY=$(type -P python3)
D=$R/migrate_2026_2; P=${1:-full1}; OUT=$D/logs/$P; mkdir -p $OUT
files() { for s in "$@"; do find gpu4pyscf/$s -name "test_*.py" -path "*/tests/*"; done | sort; }
half() { files pbc | awk -v k=$1 'NR%2==k'; }   # k=0/1: split pbc across two tiles
run_tile() { g=$1; shift; lo=$((53+(g-6)*8)); hi=$((lo+7))
  for f in "$@"; do
    n=$(echo $f | sed 's#gpu4pyscf/##;s#/tests/#_#;s#/#_#g;s#.py##'); log=$OUT/$n.log; s=$(date +%s)
    env ZE_AFFINITY_MASK=$g timeout -s KILL 1500 taskset -c $lo-$hi $PY -m pytest -q -p no:cacheprovider $f > $log 2>&1
    echo "$P $n gpu$g rc=$? t=$(( $(date +%s)-s ))s $(tail -1 $log)" >> $D/summary_$P.txt
  done
  echo "TILEDONE gpu$g $(date +%T)" >> $D/summary_$P.txt; }
run_tile 6  $(files scf gto lib sem nac solvent qmmm) &
run_tile 7  $(files dft tdscf properties mp dispersion mcscf x2c pop md geomopt fci cc) &
run_tile 8  $(files df) &
run_tile 9  $(files grad hessian) &
run_tile 10 $(half 0) &
run_tile 11 $(half 1) &
wait; echo "DONE $(date +%T)" >> $D/summary_$P.txt
