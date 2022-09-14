from dataclasses import dataclass
from typing import Tuple, Any
from omegaconf import DictConfig
import numpy
import logging
import omegaconf
import pandas as pd

from fl.datasets.memory import load_covariates, load_phenotype, load_from_pgen, get_sample_indices
from configs.phenotype_config import MEAN_PHENO_DICT, PHENO_TYPE_DICT, PHENO_NUMPY_DICT, TYPE_LOSS_DICT, \
    TYPE_METRIC_DICT


@dataclass
class X:
    train: numpy.ndarray
    val: numpy.ndarray
    test: numpy.ndarray

@dataclass
class Y:
    train: numpy.ndarray
    val: numpy.ndarray
    test: numpy.ndarray

@dataclass
class SampleIndex:
    train: numpy.ndarray
    val: numpy.ndarray
    test: numpy.ndarray


class ExperimentDataLoader:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.logger = logging.getLogger()

    def _load_phenotype(self, path: str) -> numpy.ndarray:
        phenotype = load_phenotype(path, out_type=PHENO_NUMPY_DICT[self.cfg.data.phenotype.name], encode=(self.cfg.study == 'tg'))

        if (PHENO_TYPE_DICT[self.cfg.data.phenotype.name] == 'binary'):
            assert set(phenotype) == set([1, 2])
            return phenotype - 1
        elif (PHENO_TYPE_DICT[self.cfg.data.phenotype.name] == 'continuous') & (self.cfg.data.phenotype.name in MEAN_PHENO_DICT.keys()):
            return phenotype - MEAN_PHENO_DICT[self.cfg.data.phenotype.name]
        else:
            return phenotype

    def load(self) -> Tuple[X, Y]:

        y_train = self._load_phenotype(self.cfg.data.phenotype.train)
        y_val = self._load_phenotype(self.cfg.data.phenotype.val)
        y_test = self._load_phenotype(self.cfg.data.phenotype.test)
        test_samples_limit = self.cfg.experiment.get('test_samples_limit', None)
        y = Y(y_train, y_val, y_test[:test_samples_limit])

        sample_index = self._load_sample_indices()

        assert self.cfg.experiment.include_genotype or self.cfg.experiment.include_covariates

        if self.cfg.study == 'ukb':
            if self.cfg.experiment.include_genotype and self.cfg.experiment.include_covariates:
                x = self._load_genotype_and_covariates(sample_index)
            elif self.cfg.experiment.include_genotype:
                x = self._load_genotype(sample_index)
            else:
                x = self.load_covariates()

        elif self.cfg.study == 'tg':
            x = self._load_pcs()

        else:
            raise ValueError('Please define the study in config! See src/configs/default.yaml')

        return x, y

    def _load_genotype_and_covariates(self, sample_index: SampleIndex) -> X:
        test_samples_limit = self.cfg.experiment.get('test_samples_limit', None)
        X_train = numpy.hstack((load_from_pgen(self.cfg.data.genotype.train,
                                              self.cfg.data.gwas,
                                              snp_count=self.cfg.experiment.snp_count,
                                              sample_indices=sample_index.train),
                               load_covariates(self.cfg.data.covariates.train).astype(numpy.float16)))
        X_val = numpy.hstack((load_from_pgen(self.cfg.data.genotype.val,
                                              self.cfg.data.gwas,
                                              snp_count=self.cfg.experiment.snp_count,
                                              sample_indices=sample_index.val),
                               load_covariates(self.cfg.data.covariates.val).astype(numpy.float16)))
        X_test = numpy.hstack((load_from_pgen(self.cfg.data.genotype.test,
                                              self.cfg.data.gwas,
                                              snp_count=self.cfg.experiment.snp_count,
                                              sample_indices=sample_index.test),
                               load_covariates(self.cfg.data.covariates.test)[:test_samples_limit, :].astype(numpy.float16)))

        return X(X_train, X_val, X_test)


    def _load_genotype(self, sample_index: SampleIndex) -> X:
        X_train = load_from_pgen(self.cfg.data.genotype.train,
                                      gwas_path=self.cfg.data.get('gwas', None),
                                      snp_count=self.cfg.experiment.get('snp_count', None),
                                      sample_indices=sample_index.train)
        X_val = load_from_pgen(self.cfg.data.genotype.val,
                                    gwas_path=self.cfg.data.get('gwas', None),
                                    snp_count=self.cfg.experiment.get('snp_count', None),
                                    sample_indices=sample_index.val)
        X_test = load_from_pgen(self.cfg.data.genotype.test,
                                     gwas_path=self.cfg.data.get('gwas', None),
                                     snp_count=self.cfg.experiment.get('snp_count', None),
                                     sample_indices=sample_index.test)
        return X(X_train, X_val, X_test)

    def load_covariates(self) -> X:
        test_samples_limit = self.cfg.experiment.get('test_samples_limit', None)
        X_train = load_covariates(self.cfg.data.covariates.train)
        X_val = load_covariates(self.cfg.data.covariates.val)
        X_test = load_covariates(self.cfg.data.covariates.test)[:test_samples_limit, :]
        return X(X_train, X_val, X_test)

    def _load_pcs(self) -> X:
        X_train = load_plink_pcs(path=self.cfg.data.x_reduced.train, order_as_in_file=self.cfg.data.phenotype.train).values
        X_val = load_plink_pcs(path=self.cfg.data.x_reduced.val, order_as_in_file=self.cfg.data.phenotype.val).values
        X_test = load_plink_pcs(path=self.cfg.data.x_reduced.test, order_as_in_file=self.cfg.data.phenotype.test).values
        return X(X_train, X_val, X_test)

    def _load_sample_indices(self) -> SampleIndex:
        self.logger.info("Loading sample indices")
        test_samples_limit = self.cfg.experiment.get('test_samples_limit', None)
        si_train = get_sample_indices(self.cfg.data.genotype.train,
                                                       self.cfg.data.phenotype.train)
        si_val = get_sample_indices(self.cfg.data.genotype.val,
                                                     self.cfg.data.phenotype.val)
        si_test = get_sample_indices(self.cfg.data.genotype.test,
                                                      self.cfg.data.phenotype.test,
                                                      indices_limit=test_samples_limit)
        return SampleIndex(si_train, si_val, si_test)


def load_plink_pcs(path, order_as_in_file=None):
    """ Loads PLINK's eigenvector matrix (e.g. to be used as X for TG). If @order_as_in_file is not None,
     reorder rows of the matrix to match (IID-wise) rows of the file """
    df = pd.read_csv(path, sep='\t').rename(columns={'#IID': 'IID'}).set_index('IID').filter(regex='^PC*')
    if order_as_in_file is not None:
        y = pd.read_csv(order_as_in_file, sep='\t').set_index('IID')
        assert len(df) == len(y)
        df = df.reindex(y.index)
    return df
