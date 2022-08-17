from collections import namedtuple
from typing import Dict, List
import logging
import pandas
from ukb_loader import UKBDataLoader, BinarySDLoader


def load_pheno_cov(split_ids_path: str, 
                   ukb_dataset_path: str, 
                   covariates: Dict[str, str], 
                   phenotype_name: str, 
                   phenotype_code: int,
                   phenotype_type: str) -> pandas.DataFrame:
                   
    if phenotype_type == 'real':
        loader = UKBDataLoader(ukb_dataset_path, 'split', str(phenotype_code), list(covariates.keys()))
        pheno_cov = pandas.concat((loader.load_train(), loader.load_val(), loader.load_test()))
        pheno_cov.rename({str(phenotype_code) : phenotype_name}, axis='columns', inplace=True)
    
    elif phenotype_type == 'binary':
        sd_field_code = '20002'
        loader = BinarySDLoader(ukb_dataset_path, 'split', sd_field_code, list(covariates.keys()), phenotype_code, na_as_false=False)
        pheno_cov = pandas.concat((loader.load_train(), loader.load_val(), loader.load_test()))
        pheno_cov.rename({str(sd_field_code) : phenotype_name}, axis='columns', inplace=True)
        pheno_cov.loc[:, phenotype_name] += 1 # Compatibility with PLINK's coding of case/control as 2/1
    
    else:
        raise NotImpementedError('Implemented: real and binary self-reported phenotypes')
    
    pheno_cov.rename(covariates, axis='columns', inplace=True)
    
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
    phenotype_type = snakemake.params['phenotype_type']

    pheno_cov = load_pheno_cov(split_ids_path, ukb_dataset_path, covariate_cols, phenotype_name, phenotype_code, phenotype_type)
    logging.info(f'loaded {pheno_cov.shape[0]} phenotypes for {phenotype_name} and covariates {covariate_cols}')
    pcs = load_pcs(pcs_path)
    logging.info(f'loaded {pcs.shape[0]} PCs from {pcs_path}')

    print(f'pheno_cov has columns {pheno_cov.columns}')
    only_pheno = pheno_cov.loc[:, ['FID', 'IID', phenotype_name]]
    pheno_cov.drop(phenotype_name, axis='columns', inplace=True)
    
    pcs_cov = pheno_cov.merge(pcs, on=['FID', 'IID'])
    pcs_cov = pcs_cov[['FID', 'IID'] + list(pcs_cov.columns.difference(['FID', 'IID']))]
    pcs_cov.fillna(pcs_cov.mean(), inplace=True)
    
    phenotypes_path = snakemake.output['phenotypes']
    covariates_path = snakemake.output['covariates']
    
    logging.info(f'Writing {only_pheno.shape[0]} phenotypes to {phenotypes_path}')
    only_pheno.sort_values(by='IID', axis=0, ascending=True, inplace=True)
    only_pheno.to_csv(phenotypes_path, sep='\t', index=False)
    logging.info(f'Writing {pcs_cov.shape[0]} pcs and covariates to {covariates_path}')
    pcs_cov.sort_values(by='IID', axis=0, ascending=True, inplace=True)
    pcs_cov.to_csv(covariates_path, sep='\t', index=False)
