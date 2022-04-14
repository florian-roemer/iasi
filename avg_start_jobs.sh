#!/bin/bash

years=( "2011" "2013" "2016" "2017" )
months=( "01" "03" "05" "07" "09" "11" )

for year in ${years[@]}
do
for month in ${months[@]}
do
sbatch_simple_bigmem compute2 avg2_$year'_'$month 36 00:10:00 python postprocess_iasi.py $year $month
done
done 
