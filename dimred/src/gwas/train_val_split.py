from typing import List
import hydra
from omegaconf import DictConfig

import pandas
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from utils.split import Split


FOLD_COUNT = 10


class CVSplitter:
    def __init__(self, split: Split) -> None:
        """Splits PCs, phenotypes and covariates into folds. 
        Merges PCs and covariates for GWAS. 
        Extracts phenotypes.

        Args:
            split (Split): Split object for phenotype, PC and covariates file paths manipulation
        """        
        self.split = split

    def split_ids(self, node_index: int, random_state: int):
        """
        Splits sample ids into 10-fold cv for each node. 80% are train, 10% are val and 10% are test.
        
        Args:
            node_index (int): Index of node
            random_state (int): Fixed random_state for train_test_split sklearn function
        """    
        path = self.split.get_source_ids_path(node_index)
        # we do not need sex here
        ids = pandas.read_table(path).loc[:, ['FID', 'IID']]
        train_size, val_size, test_size = int(ids.shape[0]*0.8), int(ids.shape[0]*0.1), int(ids.shape[0]*0.1)
        train_size += (ids.shape[0] - train_size - test_size - val_size)

        kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
        for fold_index, (train_val_indices, test_indices) in enumerate(kfold.split(ids.loc[:, ['FID', 'IID']])):
            
            train_indices, val_indices = train_test_split(train_val_indices, train_size=train_size, random_state=random_state)
            
            for indices, part in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
                out_path = self.split.get_ids_path(node_index, fold_index, part)
                ids.iloc[indices, :].to_csv(out_path, sep='\t', index=False)


    def split_phenotypes(self, node_index: int) -> str:
        """
        Extracts train or val subset of samples from file with phenotypes and covariates

        Args:
            node_index (int): Index of particular node

        Returns:
            str: Path to extracted subset of samples with phenotypes and covariates
        """    
        phenotype = pandas.read_table(self.split.get_source_phenotype_path(node_index))

        for fold_index in range(FOLD_COUNT):
            for part in ['train', 'val', 'test']:
                
                fold_indices = pandas.read_table(self.split.get_ids_path(node_index, fold_index, part))
                part_phenotype = phenotype.merge(fold_indices, how='inner', on=['FID', 'IID'])

                out_path = self.split.get_cov_pheno_path(node_index, fold_index, part)
                part_phenotype.to_csv(out_path, sep='\t', index=False)

        return out_path


    def split_pca(self, node_index: int) -> str:
        """
        Extracts train or val subset of samples from file with principal components
        Args:
            node_index (int): Index of particular node

        Returns:
            str: Path to extracted subset of samples with PCs
        """    
        pca = pandas.read_table(self.split.get_source_pca_path(node_index))
        pca.rename({'#FID': 'FID'}, axis='columns', inplace=True)

        for fold_index in range(FOLD_COUNT):
            for part in ['train', 'val', 'test']:
                
                fold_indices = pandas.read_table(self.split.get_ids_path(node_index, fold_index, part))
                
                fold_pca = pca.merge(fold_indices, how='inner', on=['FID', 'IID'])
                fold_pca.to_csv(self.split.get_pca_path(node_index, fold_index, part), sep='\t', index=False)


    def prepare_cov_and_phenotypes(self, node_index: int):
        """Transforms phenotype+covariates and pca files into phenotype and pca+covariates files for each fold.

        Args:
            node_index (int): Index of node
        """        
        for fold_index in range(FOLD_COUNT):
            for part in ['train', 'val', 'test']: 

                pca_path = self.split.get_pca_path(node_index, fold_index, part)
                cov_pheno_path = self.split.get_cov_pheno_path(node_index, fold_index, part)
                pca_cov_path = self.split.get_pca_cov_path(node_index, fold_index, part)
                phenotype_path = self.split.get_phenotype_path(node_index, fold_index, part)

                self.prepare_cov_and_phenotype_for_fold(pca_path, cov_pheno_path, pca_cov_path, phenotype_path)


    def prepare_cov_and_phenotype_for_fold(
            self,
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


    def standardize_covariates_and_phenotypes(self, node_index: int, covariates: List[str] = None):
        """Z-standardizes {covariates} and phenotype for each fold and particular node {node_index}

        Args:
            node_index (int): Index of node
            covariates (List[str], optional): Covariates to standardize. If None, every covariate will be standardized. Defaults to None.
        """        
        for fold_index in range(FOLD_COUNT):
            self.standardize(
                    self.split.get_pca_cov_path(node_index, fold_index, 'train'),
                    self.split.get_pca_cov_path(node_index, fold_index, 'val'),
                    self.split.get_pca_cov_path(node_index, fold_index, 'test'),
                    covariates
            )

            # standardize phenotype in phenotype-only file
            self.standardize(
                    self.split.get_phenotype_path(node_index, fold_index, 'train'),
                    self.split.get_phenotype_path(node_index, fold_index, 'val'),
                    self.split.get_phenotype_path(node_index, fold_index, 'test'),
                    None
            )


    def standardize(self, train_path: str, val_path: str, test_path: str, columns: List[str]):
        """
        Infers mean and std from columns in {train_path} and standardizes both train, test and val columns in-place
        TODO: think about non-iid data!!!

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
    
    split = Split(cfg.split_dir, cfg.phenotype.name, cfg.node_count, FOLD_COUNT)
    cv = CVSplitter(split)

    for node_index in range(cfg.node_count):
        
        print(f'Node: {node_index}')
        cv.split_ids(node_index, cfg.random_state)
        print(f'ids were splitted')

        cv.split_phenotypes(node_index)
        print(f'phenotypes were splitted')

        cv.split_pca(node_index)
        print(f'PCs were splitted')

        cv.prepare_cov_and_phenotypes(node_index)
        print(f'covariates and phenotypes were prepared')

        cv.standardize_covariates_and_phenotypes(node_index, cfg.zstd_covariates)
        print(f'covariates {cfg.zstd_covariates} and phenotype were standardized')

        print(f'splitting into {FOLD_COUNT} folds for node {node_index} completed')
        print()
        

if __name__ == '__main__':
    main()