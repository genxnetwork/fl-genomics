from typing import Tuple, List
import hydra
from omegaconf import DictConfig
import pandas
from utils.plink import run_plink
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_phenotype_path(source_dir: str, name: str, split_index: int, part_name: str = None) -> str:
    if part_name is None:
        return os.path.join(source_dir, f'{name}_split_{split_index}.csv')
    else:
        return os.path.join(source_dir, f'{name}_split_{split_index}_{part_name}.tsv')


def get_pca_cov_path(source_dir: str, name: str, split_index: int, part_name: str) -> str:
    return os.path.join(source_dir, f'{name}_split_{split_index}_{part_name}.tsv')


def split_ids(split_ids_dir: str, split_index: int, random_state: int) -> Tuple[str, str]:
    """
    Splits sample ids into train and val subsets
    
    Args:
        split_ids_dir (str): Dir where sample ids files are located
        split_index (int): Index of split part
        random_state (int): Fixed random_state for train_test_split sklearn function

    Returns:
        Tuple[str, str]: Paths to files with train sample ids and val sample ids
    """    
    path = os.path.join(split_ids_dir, f'{split_index}.csv')
    ids = pandas.read_table(path)
    fid_train, fid_val, iid_train, iid_val = train_test_split(ids.FID, ids.IID, random_state=random_state)
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


def split_genotypes(genotype_dir: str, split_index: int, threads: int, memory_mb: int, part_name: str, split_ids_path: str):
    """
    Extracts train or val subset of genotypes

    Args:
        genotype_dir (str): Dir where source genotype files are located for each {split_index}
        split_index (int): Index of particular split part
        threads (int): Number of threads for plink
        memory_mb (int): Maximum memory for plink to allocate
        part_name (str): train or val
        split_ids_path (str): Path where ids for particular split part and train or val subsets are located
    """    
    args = [
        '--pfile',
        os.path.join(genotype_dir, f'split{split_index}_filtered'),
        '--threads', str(threads), '--memory', str(memory_mb),
        '--keep', split_ids_path,
        '--make-pgen', '--out', os.path.join(genotype_dir, f'split_{split_index}_filtered_{part_name}')
    ]
    run_plink(args)


def split_phenotypes(phenotype_dir: str, phenotype_name: str, split_index: int, part_name: str, split_ids_path: str) -> str:
    """
    Extracts train or val subset of samples from file with phenotypes and covariates

    Args:
        phenotype_dir (str): Dir where source files are located for each {split_index}
        phenotype_name (str): Name of phenotype
        split_index (int): Index of particular split part
        part_name (str): train or val
        split_ids_path (str): Path where ids for particular split part and train or val subsets are located

    Returns:
        str: Path to extracted subset of samples with phenotypes and covariates
    """    
    phenotype = pandas.read_table(get_phenotype_path(phenotype_dir, phenotype_name, split_index, None))
    split_ids = pandas.read_table(split_ids_path)

    part_phenotype = phenotype.merge(split_ids, how='inner', on=['FID', 'IID'])

    out_path = get_phenotype_path(phenotype_dir, phenotype_name, split_index, part_name)
    part_phenotype.to_csv(out_path, sep='\t', index=False)
    return out_path


def split_pca(pca_dir: str, split_index: int, part_name: str, split_ids_path: str) -> str:
    """
    Extracts train or val subset of samples from file with principal components
    Args:
        pca_dir (str): Dir where source PC files are located for each split part
        split_index (int): Index of particular split part
        part_name (str): train or val
        split_ids_path (str): Path where ids for particular split part and train or val subsets are located

    Returns:
        str: Path to extracted subset of samples with PCs
    """    
    pca = pandas.read_table(os.path.join(pca_dir, f'{split_index}_projections.csv.eigenvec'))
    split_ids = pandas.read_table(split_ids_path)
    pca.rename({'#FID': 'FID'}, axis='columns', inplace=True)

    part_pca = pca.merge(split_ids, how='inner', on=['FID', 'IID'])

    out_path = os.path.join(pca_dir, f'{split_index}_projections_{part_name}.csv.eigenvec')
    part_pca.to_csv(out_path, sep='\t', index=False)
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
    pca = pandas.read_table(os.path.join(pca_dir, f'{split_index}_projections_{part_name}.csv.eigenvec'))
    cov_pheno = pandas.read_table(get_phenotype_path(pheno_cov_dir, pheno_name, split_index, part_name))
    cov_columns = list(cov_pheno.columns)[2:-1]
    pheno_column = cov_pheno.columns[-1]
    print(f'COV_PHENO_COLUMNS are: {cov_columns}')
    merged = pca.merge(cov_pheno, how='inner', on=['FID', 'IID'])
    
    pca_cov = merged.loc[:, ['FID', 'IID'] + [f'PC{i}' for i in range(1, 11)] + cov_columns]
    pca_cov.fillna(pca_cov.mean(), inplace=True)
    phenotype = merged.loc[:, ['FID', 'IID'] + [pheno_column]]

    phenotype_path = get_phenotype_path(phenotypes_dir, pheno_name, split_index, part_name)
    phenotype.to_csv(phenotype_path, sep='\t', index=False)

    pca_cov_path = get_pca_cov_path(covariates_dir, pheno_name, split_index, part_name)
    pca_cov.to_csv(pca_cov_path, sep='\t', index=False)
    
    return phenotype_path, pca_cov_path


def standardize(train_path: str, val_path: str, columns: List[str] = None):
    """
    Infers mean and std from columns in {train_path} and standardizes both train and val columns

    Args:
        train_path (str): Path to .tsv file with train data. First two columns should be FID, IID
        val_path (str): Path to .tsv file with val data. First two columns should be FID, IID
        columns (List[str], optional): List of columns to standardize. By default all columns except FID and IID will be standardized. Defaults to None. 
    """    
    train_data = pandas.read_table(train_path)
    val_data = pandas.read_table(val_path)

    scaler = StandardScaler()
    if columns is None:
        train_data.iloc[:, 2:] = scaler.fit_transform(train_data.iloc[:, 2:]) # 0,1 are FID, IID
        val_data.iloc[:, 2:] = scaler.transform(val_data.iloc[:, 2:])
    else:
        train_data.loc[:, columns] = scaler.fit_transform(train_data.loc[:, columns]) 
        val_data.loc[:, columns] = scaler.transform(val_data.loc[:, columns])

    train_data.to_csv(train_path, sep='\t', index=False)
    val_data.to_csv(val_path, sep='\t', index=False)


@hydra.main(config_path='configs', config_name='split')
def main(cfg: DictConfig):
    for split_index in range(cfg.split_count):
        train_ids, val_ids = split_ids(os.path.join(cfg.split_dir, 'split_ids'), split_index, cfg.random_state)
        cov_dir = os.path.join(cfg.split_dir, 'covariates')
        cov_pheno_dir = os.path.join(cfg.split_dir, 'phenotypes')
        pca_dir = os.path.join(cfg.split_dir, 'pca')
        phenotypes_dir = os.path.join(cfg.split_dir, 'only_phenotypes')
            
        for part_name, part_ids in zip(['train', 'val'], [train_ids, val_ids]):
            
            os.makedirs(phenotypes_dir, exist_ok=True)
            os.makedirs(cov_dir, exist_ok=True)
            
            # split genotypes into train and test val for split part {part_name}
            split_genotypes(os.path.join(cfg.split_dir, 'genotypes'), split_index, cfg.threads, cfg.memory_mb, part_name, part_ids)

            # split phenotypes and covariates into train and val for split part {part_name}
            split_phenotypes(cov_pheno_dir, cfg.phenotype.name, split_index, part_name, part_ids)

            # split file with PCs into train and val for split part {part_name}
            split_pca(pca_dir, split_index, part_name, part_ids)

            # extract covariates from cov_pheno file and merge it with PCs.
            # write phenotype-only files
            prepare_cov_and_phenotypes(pca_dir, cov_pheno_dir, cfg.phenotype.name, part_name, split_index, cov_dir, phenotypes_dir)  

            print(f'Genotypes, covariates, and phenotypes were prepared for split {split_index} and part {part_name}') 
        
        # standardize {cfg.zstd_covariates} columns in covariate files
        standardize(
            get_pca_cov_path(cov_dir, cfg.phenotype.name, split_index, 'train'),
            get_pca_cov_path(cov_dir, cfg.phenotype.name, split_index, 'val'),
            cfg.zstd_covariates
        )

        # standardize phenotype in phenotype-only file
        standardize(
            get_phenotype_path(phenotypes_dir, cfg.phenotype.name, split_index, 'train'),
            get_phenotype_path(phenotypes_dir, cfg.phenotype.name, split_index, 'val')
        )


if __name__ == '__main__':
    main()