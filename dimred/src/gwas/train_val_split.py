from typing import Tuple
import hydra
from omegaconf import DictConfig
import pandas
from utils.plink import run_plink
import shutil
import os
from sklearn.model_selection import train_test_split


def get_phenotype_path(source_dir: str, name: str, split_index: int, part_name: str = None) -> str:
    if part_name is None:
        return os.path.join(source_dir, f'{name}_split_{split_index}.csv')
    else:
        return os.path.join(source_dir, f'{name}_split_{split_index}_{part_name}.tsv')


def split_ids(split_ids_dir: str, split_index: int, random_state: int) -> Tuple[str, str]:
    path = os.path.join(split_ids_dir, f'{split_index}.csv')
    ids = pandas.read_table(path)
    fid_train, fid_val, iid_train, iid_val = train_test_split(ids.fid, ids.iid, random_state=random_state)
    train = pandas.DataFrame()
    train.loc[:, 'FID'] = fid_train
    train.loc[:, 'IID'] = iid_train
    val = pandas.DataFrame()
    val.loc[:, 'FID'] = fid_val
    val.loc[:, 'IID'] = iid_val
    train_out_path = os.path.join(split_ids_dir, f'{split_index}_train.tsv')
    val_out_path = os.path.join(split_ids_dir, f'{split_index}_val.tsv')
    train.to_csv(train_out_path, sep='\t', index=False)
    val.to_csv(val_out_path, sep='\t', index=False)    
    return train_out_path, val_out_path


def split_genotypes(genotype_dir: str, split_index: int, threads: int, part_name: str, split_ids_path: str):
    args = [
        '--pfile',
        os.path.join(genotype_dir, f'split{split_index}_filtered'),
        '--threads', str(threads),
        '--keep', split_ids_path,
        '--make-pgen', '--out', f'split{split_index}_filtered_{part_name}'
    ]
    run_plink(args)


def split_phenotypes(phenotype_dir: str, phenotype_name: str, split_index: int, part_name: str, split_ids_path: str) -> str:
    phenotype = pandas.read_table(get_phenotype_path(phenotype_dir, phenotype_name, split_index, part_name))
    split_ids = pandas.read_table(split_ids_path)

    part_phenotype = phenotype.merge(split_ids, how='inner', on=['FID', 'IID'])

    out_path = get_phenotype_path(phenotype_dir, phenotype_name, split_index, part_name)
    part_phenotype.reset_index(inplace=True)
    part_phenotype.to_csv(out_path, sep='\t')
    return out_path


def split_pca(pca_dir: str, split_index: int, part_name: str, split_ids_path: str) -> str:
    pca = pandas.read_table(os.path.join(pca_dir, f'{split_index}_projections.eigenvec'))
    split_ids = pandas.read_table(split_ids_path)
    pca.rename({'#FID', 'FID'}, axis='columns', inplace=True)

    part_pca = pca.merge(split_ids, how='inner', on=['FID', 'IID'])

    out_path = os.path.join(pca_dir, f'{split_index}_projections_{part_name}.eigenvec')
    part_pca.reset_index(inplace=True)
    part_pca.to_csv(out_path, sep='\t')
    return out_path


def prepare_cov_and_phenotypes(pca_dir: str, 
                               pheno_cov_dir: str, 
                               pheno_name: str,
                               part_name: str, 
                               split_index: int, 
                               covariates_dir: str, 
                               phenotypes_dir: str) -> Tuple[str, str]:
    """Transforms phenotype+covariates and pca files into phenotype and pca+covariates files.
    This is required by plink 2.0 --glm command

    Args:
        pca_path (str): Path to PCA eigenvec file computed by plink 2.0
        phenotype_path (str): Path to phenotype and covariates file prepared with ukb_loader

    Returns:
        Tuple[str, str]: Paths to phenotype file and to pca+covariates file
    """   
    pca = pandas.read_table(os.path.join(pca_dir, f'{split_index}_projections_{part_name}.eigenvec'))
    cov_pheno = pandas.read_table(get_phenotype_path(pheno_cov_dir, pheno_name, split_index, part_name))
    cov_columns = list(cov_pheno.columns)[2:-1]
    pheno_column = cov_pheno.columns[-1]

    merged = pca.merge(cov_pheno, how='inner', on=['FID', 'IID'])
    
    pca_cov = merged.loc[:, [f'PC{i}' for i in range(1, 11)] + cov_columns]
    phenotype = merged.loc[:, [pheno_column]]

    phenotype_path = get_phenotype_path(phenotypes_dir, pheno_name, split_index, part_name)
    phenotype.reset_index().to_csv(phenotype_path, sep='\t', index=False)

    pca_cov_path = os.path.join(covariates_dir, f'{pheno_name}_split_{split_index}_{part_name}.tsv')
    pca_cov.reset_index().to_csv(pca_cov_path, sep='\t', index=False)
    
    return phenotype_path, pca_cov_path


@hydra.main(config_path='configs', config_name='split')
def main(cfg: DictConfig):

    for split_index in range(cfg.split_count):
        train_ids, val_ids = split_ids(os.path.join(cfg.split_dir, 'split_ids'), split_index)

        for part_name, part_ids in zip(['train', 'val'], [train_ids, val_ids]):
            cov_pheno_dir = os.path.join(cfg.split_dir, 'phenotypes')
            pca_dir = os.path.join(cfg.split_dir, 'pca')
            phenotypes_dir = os.path.join(cfg.split_dir, 'only_phenotypes')
            cov_dir = os.path.join(cfg.split_dir, 'covariates')

            split_genotypes(os.path.join(cfg.split_dir, 'genotypes'), split_index, cfg.threads, part_name, part_ids)

            split_phenotypes(cov_pheno_dir, cfg.phenotype.name, split_index, part_name, part_ids)

            split_pca(pca_dir, split_index, part_name, part_ids)
        
            prepare_cov_and_phenotypes(pca_dir, cov_pheno_dir, cfg.phenotype.name, part_name, split_index, cov_dir, phenotypes_dir)  

            print(f'Covariates and phenotypes were prepared for split {split_index} and part {part_name}') 


if __name__ == '__main__':
    main()