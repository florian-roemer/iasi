#!/bin/bash
year=$1
month=$2

scripts='/DSNNAS/Repro_Temp/users/vijuj/git/iasi'

for day in {1..9}
do
sbatch_simple_bigmem compute2 IASI$year$month'0'$day 36 01:30:00 python $scripts/process_iasi.py $year $month '0'$day
done

for day in {10..31}
do
sbatch_simple_bigmem compute2 IASI$year$month$day 36 01:30:00 python $scripts/process_iasi.py $year $month $day
done
