#!/bin/bash
#year=$1
#month=$2

#scripts='/DSNNAS/Repro_Temp/users/vijuj/git/iasi/'
#data='/DSNNAS/Repro/iasi/IASI_eum_fcdr_r0100/data/M02/level1c/iasi/native'

data='/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test'

domain1=( "global" "tropics")
domain2=( "all-sky" "clear-sky")
domain3=( "land+ocean" "ocean-only")

for dom1 in ${domain1[@]}
do
for dom2 in ${domain2[@]}
do
for dom3 in ${domain3[@]}
do
for year in {2007..2021}
do
mkdir -p $data/$dom1/$dom2/$dom3/$year
done
done
done
done

#for year in {2007..2021}
#do
#mkdir -p $data/$year
#done

#for day in {1..9}
#do
#sbatch_simple_bigmem compute2 IASI$year$month'0'$day 36 00:30:00 python $filepath'process_iasi.py' $year $month '0'$day
#done

#for day in {10..31}
#do
#sbatch_simple_bigmem compute2 IASI$year$month$day 36 00:30:00 python $filepath'process_iasi.py' $year $month $day
#done
