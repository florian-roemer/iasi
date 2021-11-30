#!/bin/bash

data='/DSNNAS/Repro_Temp/users/vijuj/data/iasi_flux'

domain1=( "global" "tropics" "extra" )
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
for month in {1..9}
do
mkdir -p $data/$dom1/$dom2/$dom3/$year/'0'$month
done
for month in {10..12}
do
mkdir -p $data/$dom1/$dom2/$dom3/$year/$month
done
done
done
done
