import os
import pandas as pd
import numpy as np

from utils.loaders import X, Y
from nn.lightning import DataModule


class DataProvider:
    def __init__(self, pca_directory, ancestry_directory, num_components=None, normalize_std=True):
        """
        Data provider relies on the following structure of the directories. PCA directory provides `X`
        matrices for training while `y` values (classes) are taken from the ancestry directory.

        <pca_directory>
        |-> AFR
        |   |-> fold_0_test_projections.csv.eigenvec.sscore
        |   |-> fold_0_train_projections.csv.eigenvec.sscore
        |   |-> fold_0_val_projections.csv.eigenvec.sscore
        |   |-> ...
        |-> AMR
        |-> ...

        <ancestry_directory>
        |-> AFR
        |   |-> fold_0_train.tsv
        |   |-> fold_0_test.tsv
        |   |-> fold_0_val.tsv
        |   |-> ...
        |-> AMR
        |-> ...
        """

        self.pca_directory = pca_directory
        self.ancestry_directory = ancestry_directory
        self.num_components = num_components
        self.normalize_std = normalize_std

    def get_pca_file(self, node, fold, part):
        return os.path.join(
            self.pca_directory,
            f'{node}/fold_{fold}_{part}_projections.csv.eigenvec.sscore'
        )

    def get_ancestry_file(self, node, fold, part):
        return os.path.join(self.ancestry_directory, f'{node}/fold_{fold}_{part}.tsv')

    def load_data(self, node, fold, part):
        """
        Load data: `X` = plink PCs, `y` = ancestry.
        """

        # Drop 2nd (ALLELE_CT) and 3rd (NAMED_ALLELE_DOSAGE_SUM), columns of the plink PCs file
        X = pd.read_csv(self.get_pca_file(node, fold, part), sep='\t', header=0)
        X = X.drop(X.columns[1:3], axis=1).rename(columns={'#IID': 'IID'}).set_index('IID')
        y = pd.read_csv(self.get_ancestry_file(node, fold, part), sep='\t', header=0).set_index('IID')

        # Ensure that the order of ids is consistent
        assert len(X) == len(y)
        X = X.reindex(y.index)

        _, y_encoded = np.unique(y['ancestry'].to_numpy(), return_inverse=True)
        if self.num_components is None:
            return X.to_numpy(), y_encoded
        else:
            return X.to_numpy()[:, :self.num_components], y_encoded

    def load_train_data(self, node, fold):
        return self.load_data(node, fold, 'train')

    def load_validation_data(self, node, fold):
        return self.load_data(node, fold, 'val')

    def load_test_data(self, node, fold):
        return self.load_data(node, fold, 'test')

    def create_data_module(self, node, fold):
        """
        Creates data module for training.
        """

        X_train, y_train = self.load_train_data(node, fold)
        X_validation, y_validation = self.load_validation_data(node, fold)
        X_test, y_test = self.load_test_data(node, fold)

        if self.normalize_std:
            X_train_std = X_train.std(axis=0)
            X_validation_std = X_validation.std(axis=0)
            X_test_std = X_test.std(axis=0)

            X_validation = X_validation * (X_train_std / X_validation_std)
            X_test = X_test * (X_train_std / X_test_std)

        return DataModule(
            X(X_train, X_validation, X_test),
            Y(y_train, y_validation, y_test),
            batch_size=len(X_train)
        )
