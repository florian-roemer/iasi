#!/bin/bash

data='/work/um0878/user_data/froemer/rare_mistral/data/IASI/final'

domain1=( "global" "tropics" "extra" )
domain2=( "all-sky" "clear-sky" )
domain3=( "land+ocean" "ocean-only" ) 

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
echo $data/$dom1/$dom2/$dom3/$year/'0'$month
mkdir -p $data/$dom1/$dom2/$dom3/$year/'0'$month
done
for month in {10..12}
do
echo $data/$dom1/$dom2/$dom3/$year/$month
mkdir -p $data/$dom1/$dom2/$dom3/$year/$month
done
done
done
done
done

