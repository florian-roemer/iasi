#!/bin/bash

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
