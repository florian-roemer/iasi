#!/bin/bash
year=$1
month=$2

filepath='/mnt/lustre01/pf/zmaw/u301023/iasi'

for day in {1..9}
do
sbatch_simple_bigmem compute2 IASI$year$month'0'$day 36 02:00:00 python $filepath/test_process_iasi.py $year $month '0'$day
done

for day in {10..31}
do
sbatch_simple_bigmem compute2 IASI$year$month$day 36 02:00:00 python $filepath/test_process_iasi.py $year $month $day
done
