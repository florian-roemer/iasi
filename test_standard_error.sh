#!/bin/bash
year=$1
month=$2

filepath='/mnt/lustre01/pf/zmaw/u301023/iasi'

years=( 2011 2013 2016 2017 )

for year in ${years[@]}
do
for day in {14..28..14}
do
for month in {1..9..2}
do
sbatch_simple_bigmem compute2 save_std$year'0'$month$day'0'$month 36 00:15:00 python $filepath/test_process_iasi.py $year '0'$month $day '0'$month
done
sbatch_simple_bigmem compute2 save_std$year'11'$day'11' 36 00:15:00 python $filepath/test_process_iasi.py $year 11 $day 11
done
done
