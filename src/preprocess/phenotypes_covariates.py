from collections import namedtuple
from typing import List
import logging
import pandas


def load_pheno_cov(split_ids_path: str, 
                   ukb_dataset_path: str, 
                   covariates: List[str], 
                   phenotype_name: str, 
                   phenotype_code: int) -> pandas.DataFrame:
                   
                   pass

def load_pcs(pcs_path: str) -> pandas.DataFrame:
    
    pass


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
    pcs_path = snakemake.output['pca']
    ukb_dataset_path = snakemake.input['ukb_dataset']

    covariate_cols = snakemake.params['covariates']
    phenotype_name = snakemake.params['phenotype_name']
    phenotype_code = snakemake.params['phenotype_code']

    only_pheno_path = snakemake.output['phenotype']
    pcs_cov_path = snakemake.output['covariates']

    pheno_cov = load_pheno_cov(split_ids_path, ukb_dataset_path, covariate_cols, phenotype_name, phenotype_code)
    pcs = load_pcs(pcs_path)

    only_pheno = pheno_cov.loc[:, [[phenotype_name]]]
    only_pheno.to_csv()
    pheno_cov.drop(phenotype_name, axis='columns', inplace=True)
    
    pcs_cov = pheno_cov.merge(pcs, on=['FID', 'IID'])
    
    phenotypes = snakemake.output['phenotypes']
    covariates = snakemake.output['covariates']
    
