#!/bin/bash

python population_qc.py
python split.py

# British split
for i in {0..4}
do
  plink2 --bfile /gpfs/gpfs0/ukb_data/plink/plink --make-pgen --out /gpfs/gpfs0/ukb_data/fml/british_split/genotypes/split${i} --keep /gpfs/gpfs0/ukb_data/fml/british_split/split_ids/${i}.csv
done

# Ethnic split 
for i in {0..5}
do
  plink2 --bfile /gpfs/gpfs0/ukb_data/plink/plink --make-pgen --out /gpfs/gpfs0/ukb_data/fml/ethnic_split/genotypes/split${i} --keep /gpfs/gpfs0/ukb_data/fml/ethnic_split/split_ids/${i}.csv
done

