#!/bin/bash

#data='/scratch/uni/u237/user_data/froemer/iasi/data/'
data='avg_error'
domain1=( "global" "tropics" "extra" )
domain2=( "all-sky" "clear-sky" )
domain3=( "land+ocean" "ocean-only" ) 

for dom1 in ${domain1[@]}
do
for dom2 in ${domain2[@]}
do
for dom3 in ${domain3[@]}
do
echo $data/$dom1/$dom2/$dom3
mkdir -p $data/$dom1/$dom2/$dom3
done
done
done

