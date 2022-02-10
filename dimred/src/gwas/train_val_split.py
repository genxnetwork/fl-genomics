from typing import Tuple, List
import hydra
from omegaconf import DictConfig
import pandas
from utils.plink import run_plink
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler


FOLD_COUNT = 10


def get_phenotype_path(source_dir: str, name: str, node_index: int, fold_index: int = None, part: str = None) -> str:
    if fold_index is None or part is None:
        # TODO: rename split to node in preprocessing
        return os.path.join(source_dir, f'{name}_split_{node_index}.csv')
    else:
        return os.path.join(source_dir, name, f'node_{node_index}', f'fold_{fold_index}_{part}.tsv')

def ensure_cov_pheno_dir(cov_pheno_dir: str, phenotype_name: str, node_index: int):
    os.makedirs(os.path.join(cov_pheno_dir, phenotype_name, f'node_{node_index}'), exist_ok=True)


def get_pca_cov_path(source_dir: str, name: str, node_index: int, fold_index: int, part: str) -> str:
    return os.path.join(source_dir, f'{name}_node_{node_index}_fold_{fold_index}_{part}.tsv')

def get_pca_path(pca_dir: str, node_index: int, fold_index: int = None, part: str = None) -> str:
    if fold_index is None or part is None:
        return os.path.join(pca_dir, f'{node_index}_projections.csv.eigenvec')
    else:
        return os.path.join(pca_dir, f'node_{node_index}', f'fold_{fold_index}_projections_{part}.csv.eigenvec')

def ensure_pca_dir(pca_dir: str, node_index: int):
    os.makedirs(os.path.join(pca_dir, f'node_{node_index}'), exist_ok=True)


def ensure_phenotypes_only_dir(phenotypes_dir: str, phenotype_name: str, node_index: int):
    os.makedirs(os.path.join(phenotypes_dir, phenotype_name, f'node_{node_index}'), exist_ok=True)


def get_ids_path(split_ids_dir: str, node_index: int, fold_index: int, part_name: str) -> str:
    return os.path.join(split_ids_dir, f'node_{node_index}', f'fold_{fold_index}_{part_name}.tsv')


def split_ids(split_ids_dir: str, node_index: int, random_state: int):
    """
    Splits sample ids into 10-fold cv for each node. 80% are train, 10% are val and 10% are test.
    
    Args:
        split_ids_dir (str): Dir where sample ids files are located
        node_index (int): Index of node
        random_state (int): Fixed random_state for train_test_split sklearn function
    """    
    path = os.path.join(split_ids_dir, f'{node_index}.csv')
    # we do not need sex here
    ids = pandas.read_table(path).loc[:, ['FID', 'IID']]
    train_size, val_size, test_size = int(ids.shape[0]*0.8), int(ids.shape[0]*0.1), int(ids.shape[0]*0.1)
    train_size += (ids.shape[0] - train_size - test_size - val_size)

    kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
    for i, (train_val_indices, test_indices) in enumerate(kfold.split(ids.loc[:, ['FID', 'IID']])):
        
        train_indices, val_indices = train_test_split(train_val_indices, train_size=train_size, random_state=random_state)
        
        for indices, part in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
            out_path = get_ids_path(split_ids_dir, node_index, i, part)
            ids.iloc[indices, :].to_csv(out_path, sep='\t', index=False)


def split_phenotypes(cov_pheno_dir: str, phenotype_name: str, split_ids_dir: str, node_index: int) -> str:
    """
    Extracts train or val subset of samples from file with phenotypes and covariates

    Args:
        cov_pheno_dir (str): Dir where source files are located for each {node_index}
        phenotype_name (str): Name of phenotype
        split_ids_dir (str): Dir where ids are
        node_index (int): Index of particular node

    Returns:
        str: Path to extracted subset of samples with phenotypes and covariates
    """    
    phenotype = pandas.read_table(get_phenotype_path(cov_pheno_dir, phenotype_name, node_index, None, None))

    for fold_index in range(FOLD_COUNT):
        for part in ['train', 'val', 'test']:
            fold_indices = pandas.read_table(get_ids_path(split_ids_dir, node_index, fold_index, part))
            part_phenotype = phenotype.merge(fold_indices, how='inner', on=['FID', 'IID'])
            ensure_cov_pheno_dir(cov_pheno_dir, phenotype_name, node_index)
            out_path = get_phenotype_path(cov_pheno_dir, phenotype_name, node_index, fold_index, part)
            part_phenotype.to_csv(out_path, sep='\t', index=False)

    return out_path


def split_pca(pca_dir: str, node_index: int, split_ids_dir: str) -> str:
    """
    Extracts train or val subset of samples from file with principal components
    Args:
        pca_dir (str): Dir where source PC files are located for each node
        node_index (int): Index of particular node
        split_ids_dir (str): Dir where ids for particular node for all folds are located

    Returns:
        str: Path to extracted subset of samples with PCs
    """    
    pca = pandas.read_table(os.path.join(pca_dir, f'{node_index}_projections.csv.eigenvec'))
    pca.rename({'#FID': 'FID'}, axis='columns', inplace=True)

    for fold_index in range(FOLD_COUNT):
        for part in ['train', 'val', 'test']:
            fold_indices = pandas.read_table(get_ids_path(split_ids_dir, node_index, fold_index, part))
            fold_pca = pca.merge(fold_indices, how='inner', on=['FID', 'IID'])
            ensure_pca_dir(pca_dir, node_index)
            fold_pca.to_csv(get_pca_path(pca_dir, node_index, fold_index, part), sep='\t', index=False)


def prepare_cov_and_phenotypes(pca_dir: str, cov_pheno_dir: str, covariates_dir: str, phenotypes_dir: str, phenotype_name: str, node_index: int):
    for fold_index in range(FOLD_COUNT):
        for part in ['train', 'val', 'test']: 

            pca_path = get_pca_path(pca_dir, node_index, fold_index, part)
            cov_pheno_path = get_phenotype_path(cov_pheno_dir, phenotype_name, node_index, fold_index, part)
            pca_cov_path = get_pca_cov_path(covariates_dir, phenotype_name, node_index, fold_index, part)
            phenotype_path = get_phenotype_path(phenotypes_dir, phenotype_name, node_index, fold_index, part)
            ensure_phenotypes_only_dir(phenotypes_dir, phenotype_name, node_index)

            prepare_cov_and_phenotype_for_fold(pca_path, cov_pheno_path, pca_cov_path, phenotype_path)


def prepare_cov_and_phenotype_for_fold(
        pca_path: str,
        cov_pheno_path: str,
        pca_cov_path: str,
        phenotype_path: str
    ):

    """Transforms phenotype+covariates and pca files into phenotype and pca+covariates files.
    This is required by plink 2.0 --glm command

    Args:
        pca_path (str): Path to PCA eigenvec file computed by plink 2.0
        cov_pheno_path (str): Path to phenotype and covariates file prepared with ukb_loader
        pca_cov_path (str): Path where all covariates including PCs will be stored
        phenotype_path (str): Path with only phenotype data

    """   
    pca = pandas.read_table(pca_path)
    cov_pheno = pandas.read_table(cov_pheno_path)
    cov_columns = list(cov_pheno.columns)[2:-1]
    pheno_column = cov_pheno.columns[-1]
    # print(f'COV_PHENO_COLUMNS are: {cov_columns}')
    merged = pca.merge(cov_pheno, how='inner', on=['FID', 'IID'])
    
    pca_cov = merged.loc[:, ['FID', 'IID'] + [f'PC{i}' for i in range(1, 11)] + cov_columns]
    pca_cov.fillna(pca_cov.mean(), inplace=True)
    phenotype = merged.loc[:, ['FID', 'IID'] + [pheno_column]]

    phenotype.to_csv(phenotype_path, sep='\t', index=False)
    pca_cov.to_csv(pca_cov_path, sep='\t', index=False)


def standardize_covariates_and_phenotypes(covariates_dir: str, phenotypes_dir: str, phenotype_name: str, node_index: int, covariates: List[str] = None):
    for fold_index in range(FOLD_COUNT):
        # standardize {cfg.zstd_covariates} columns in covariate files
        standardize(
                get_pca_cov_path(covariates_dir, phenotype_name, node_index, fold_index, 'train'),
                get_pca_cov_path(covariates_dir, phenotype_name, node_index, fold_index, 'val'),
                get_pca_cov_path(covariates_dir, phenotype_name, node_index, fold_index, 'test'),
                covariates
        )

            # standardize phenotype in phenotype-only file
        standardize(
                get_phenotype_path(phenotypes_dir, phenotype_name, node_index, fold_index, 'train'),
                get_phenotype_path(phenotypes_dir, phenotype_name, node_index, fold_index, 'val'),
                get_phenotype_path(phenotypes_dir, phenotype_name, node_index, fold_index, 'test'),
                None
        )


def standardize(train_path: str, val_path: str, test_path: str, columns: List[str]):
    """
    Infers mean and std from columns in {train_path} and standardizes both train, test and val columns in-place

    Args:
        train_path (str): Path to .tsv file with train data. First two columns should be FID, IID
        val_path (str): Path to .tsv file with val data. First two columns should be FID, IID
        columns (List[str]): List of columns to standardize. By default all columns except FID and IID will be standardized. 
    """    
    train_data = pandas.read_table(train_path)
    val_data = pandas.read_table(val_path)
    test_data = pandas.read_table(test_path)

    scaler = StandardScaler()
    if columns is None:
        train_data.iloc[:, 2:] = scaler.fit_transform(train_data.iloc[:, 2:]) # 0,1 are FID, IID
        val_data.iloc[:, 2:] = scaler.transform(val_data.iloc[:, 2:])
        test_data.iloc[:, 2:] = scaler.transform(test_data.iloc[:, 2:])

    else:
        train_data.loc[:, columns] = scaler.fit_transform(train_data.loc[:, columns]) 
        val_data.loc[:, columns] = scaler.transform(val_data.loc[:, columns])
        test_data.loc[:, columns] = scaler.transform(test_data.loc[:, columns])

    train_data.to_csv(train_path, sep='\t', index=False)
    val_data.to_csv(val_path, sep='\t', index=False)
    test_data.to_csv(test_path, sep='\t', index=False)


@hydra.main(config_path='configs', config_name='split')
def main(cfg: DictConfig):
    for node_index in range(cfg.node_count):
        split_ids_dir = os.path.join(cfg.split_dir, 'split_ids')
        os.makedirs(os.path.join(split_ids_dir, f'node_{node_index}'), exist_ok=True)

        split_ids(split_ids_dir, node_index, cfg.random_state)
        
        cov_dir = os.path.join(cfg.split_dir, 'covariates')
        cov_pheno_dir = os.path.join(cfg.split_dir, 'phenotypes')
        pca_dir = os.path.join(cfg.split_dir, 'pca')
        phenotypes_dir = os.path.join(cfg.split_dir, 'only_phenotypes')

        os.makedirs(phenotypes_dir, exist_ok=True)
        os.makedirs(cov_dir, exist_ok=True)

        split_phenotypes(cov_pheno_dir, cfg.phenotype.name, split_ids_dir, node_index)

        split_pca(pca_dir, node_index, split_ids_dir)

        prepare_cov_and_phenotypes(pca_dir, cov_pheno_dir, cov_dir, phenotypes_dir, cfg.phenotype.name, node_index)
        
        standardize_covariates_and_phenotypes(cov_dir, phenotypes_dir, cfg.phenotype.name, node_index, cfg.zstd_covariates)

        print(f'splitting into {FOLD_COUNT} folds for node {node_index} completed')
        

if __name__ == '__main__':
    main()