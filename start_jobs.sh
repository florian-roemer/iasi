#!/bin/bash
year=$1
month=$2

scripts='/work/um0878/user_data/froemer/rare_mistral/scripts/eumetsat'

for year in {2007..2021}
do
mkdir -p $data/$year
done

for day in {1..9}
do
sbatch_simple_bigmem compute2 IASI$year$month'0'$day 36 00:30:00 python $filepath'process_iasi.py' $year $month '0'$day
done

for day in {10..31}
do
sbatch_simple_bigmem compute2 IASI$year$month$day 36 00:30:00 python $filepath'process_iasi.py' $year $month $day
done
