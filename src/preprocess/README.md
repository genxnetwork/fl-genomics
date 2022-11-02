# Initial split of UK Biobank data into "node" datasets

## Datasets

0. British                          429653
1. Indian + Pakistani + Bangladeshi 7618
2. African + Caribbean              7491
3. Chinese                          1503
4. Others and Mixed                 38099

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

## Preprocessing

Our preprocessing pipeline for the assessment center split consists of the following steps.

Sample QC:
  in: genotype (all samples and variants)
  out: list of samples that pass QC
  qc steps: --mind, --king-cutoff (second degree relatives)

Splitting based on assessment centers:
  in: list of samples that passed QC, UKB assessment center data field
  out: split_ids for each node

for each node:
    Variant QC:
        in: Unfiltered genotype
        out: Filtered genotype
        params: MAF cutoff 0.05, missing genotype rate cutoff: 0.02

    Cross validation split:
        in: sample IDs
        out: train/val/test IDs

    for each CV fold:
        PCA (train):
            in: Filtered train genotype
            out: PCA eigenvecs

        PCA (project):
            in: Filtered train/val/test genotype
            out: Projected train/val/test PCs

        for each phenotype:
            Load phenotype
            Load covariates
            Normalize covariates
            GWAS:
                in: train phenotype, genotype, covariates, PCs
                out: GWAS report

for each phenotype, CV fold:
    Meta analysis:
        in: GWAS reports for all nodes
        out: Meta analysis report

