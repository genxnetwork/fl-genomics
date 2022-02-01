# Initial split of UK Biobank data into "node" datasets

## Datasets

1. British                          429653
2. Indian + Pakistani + Bangladeshi 7618
3. African + Caribbean              7491
4. Chinese                          1503
5. Others and Mixed                 38099

## Ethnicity Stats

British                       429653
Any other white background     15734
Irish                          12701
Indian                          5655
Other ethnic group              4350
Caribbean                       4289
African                         3202
Any other Asian background      1747
Pakistani                       1742
Chinese                         1503
Any other mixed background       995
White and Asian                  802
White and Black Caribbean        596
White                            542
White and Black African          401
Bangladeshi                      221
Any other Black background       118
Mixed                             46
Asian or Asian British            41
Black or Black British            26

## Data directory structure
```

$data_root
|- pca/...
|- figures/...
|- $split_name -+- genotypes/...
|               +- phenotypes/...
|               +- split_ids/...
|- valid_ids.csv  


```

