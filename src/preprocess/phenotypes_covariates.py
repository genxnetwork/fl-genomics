from collections import namedtuple
from typing import List
import logging
import pandas
from ukb_loader import UKBDataLoader


def load_pheno_cov(split_ids_path: str, 
                   ukb_dataset_path: str, 
                   covariates: List[str], 
                   phenotype_name: str, 
                   phenotype_code: int) -> pandas.DataFrame:
                   
    loader = UKBDataLoader(ukb_dataset_path, 'split', str(phenotype_code), covariates)
    pheno_cov = pandas.concat((loader.load_train(), loader.load_val(), loader.load_test()))
    pheno_cov.rename({str(phenotype_code) : phenotype_name}, axis='columns', inplace=True)
    pheno_cov.loc[:, 'FID'] = pheno_cov.index
    pheno_cov.loc[:, 'IID'] = pheno_cov.index
    pheno_cov = pheno_cov.loc[~pandas.isna(pheno_cov[phenotype_name]), :]
    split_ids = pandas.read_table(split_ids_path).loc[:, ['FID', 'IID']]
    
    return pheno_cov.merge(split_ids, how='inner', left_on=['FID', 'IID'], right_on=['FID', 'IID'])


def load_pcs(pcs_path: str) -> pandas.DataFrame:
    pcs = pandas.read_table(pcs_path)
    pcs.rename({'#FID': 'FID'}, axis='columns', inplace=True)
    pc_columns = ['FID', 'IID'] + [col for col in pcs.columns if col.startswith('PC')]
    return pcs.loc[:, pc_columns]


if __name__ == '__main__':
    try:
        snakemake
    except NameError:
        # for isolated testing
        Snakemake = namedtuple('Snakemake', ['input', 'output', 'params', 'resources', 'log'])
        snakemake = Snakemake(
            input={'name': 'test.input'},
            output={'name': 'test.output'},
            params={'parameter': 'value'},
            resources={'resource_name': 'value'},
            log=['test.log']
        )

    logging.basicConfig(filename=snakemake.log[0], level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

    split_ids_path = snakemake.input['split_ids']
    pcs_path = snakemake.input['pca']
    ukb_dataset_path = snakemake.input['dataset']

    covariate_cols = snakemake.params['covariates']
    phenotype_name = snakemake.params['phenotype_name']
    phenotype_code = snakemake.params['phenotype_code']

    pheno_cov = load_pheno_cov(split_ids_path, ukb_dataset_path, covariate_cols, phenotype_name, phenotype_code)
    logging.info(f'loaded {pheno_cov.shape[0]} phenotypes for {phenotype_name} and covariates {covariate_cols}')
    pcs = load_pcs(pcs_path)
    logging.info(f'loaded {pcs.shape[0]} PCs from {pcs_path}')

    print(f'pheno_cov has columns {pheno_cov.columns}')
    only_pheno = pheno_cov.loc[:, ['FID', 'IID', phenotype_name]]
    pheno_cov.drop(phenotype_name, axis='columns', inplace=True)
    
    pcs_cov = pheno_cov.merge(pcs, on=['FID', 'IID'])
    pcs_cov = pcs_cov[['FID', 'IID'] + list(pcs_cov.columns.difference(['FID', 'IID']))]
    
    phenotypes_path = snakemake.output['phenotypes']
    covariates_path = snakemake.output['covariates']
    
    logging.info(f'Writing {only_pheno.shape[0]} phenotypes to {phenotypes_path}')
    only_pheno.to_csv(phenotypes_path, sep='\t', index=False)
    logging.info(f'Writing {pcs_cov.shape[0]} pcs and covariates to {covariates_path}')
    pcs_cov.to_csv(covariates_path, sep='\t', index=False)