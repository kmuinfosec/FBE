#!/bin/bash
cd "$(dirname "$0")"
M=${1:-lgbm}
J=${2:-5}
mkdir -p outputs/logs
{
  for ds in kmu_tls cipherspectrum cesnet_tls22; do
    for i in ch_included ch_burst_removed drop4 drop8 fbe; do echo "$M $ds $i 0"; done
  done
  for s in 0 1 2 3 4 5 6 7 8 9; do
    for i in ch_included ch_burst_removed drop4 drop8 fbe; do echo "$M echpair $i $s"; done
  done
} | xargs -P "$J" -L 1 sh -c 'python main.py run "$0" "$1" "$2" "$3" > "outputs/logs/$0_$1_$2_$3.log" 2>&1'
python main.py summary "$M"
