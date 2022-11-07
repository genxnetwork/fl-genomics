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

    def astype(self, new_type):
        new_y = Y(
            train=self.train.astype(new_type),
            val=self.val.astype(new_type),
            test=self.test.astype(new_type)
        )
        return new_y

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
        if numpy.isnan(phenotype).sum() > 0:
            raise ValueError(f'There are {numpy.isnan(phenotype).sum()} nan values in phenotype from {path}')
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

        assert self.cfg.experiment.include_genotype or self.cfg.experiment.include_covariates

        if self.cfg.study == 'ukb':
            sample_index = self._load_sample_indices()
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

    def _get_snp_count(self):
        pvar_path = self.cfg.data.genotype + '.pvar'
        pvar = pd.read_table(pvar_path)
        return pvar.shape[0]

    def _load_genotype_and_covariates(self, sample_index: SampleIndex) -> X:
        load_strategy = self.cfg.data.get('load_strategy', 'default')
        if load_strategy == 'default':
            gwas_path = self.cfg.data.gwas
            snp_count = self.cfg.experiment.snp_count
        elif load_strategy == 'union':
            # we do not need gwas file
            # SNPs were already selected and written to a separate genotype file
            gwas_path = None
            snp_count = self._get_snp_count()
        else:
            raise ValueError(f'load_strategy should be one of ["default", "union"]')

        test_samples_limit = self.cfg.experiment.get('test_samples_limit', None)
        X_train = numpy.hstack((load_from_pgen(self.cfg.data.genotype,
                                               gwas_path,
                                               snp_count=snp_count,
                                               sample_indices=sample_index.train),
                               load_covariates(self.cfg.data.covariates.train).astype(numpy.float16)))
        X_val = numpy.hstack((load_from_pgen(self.cfg.data.genotype,
                                             gwas_path,
                                             snp_count=snp_count,
                                             sample_indices=sample_index.val),
                               load_covariates(self.cfg.data.covariates.val).astype(numpy.float16)))
        X_test = numpy.hstack((load_from_pgen(self.cfg.data.genotype,
                                              gwas_path,
                                              snp_count=snp_count,
                                              sample_indices=sample_index.test),
                               load_covariates(self.cfg.data.covariates.test)[:test_samples_limit, :].astype(numpy.float16)))

        return X(X_train, X_val, X_test)


    def _load_genotype(self, sample_index: SampleIndex) -> X:
        X_train = load_from_pgen(self.cfg.data.genotype,
                                      gwas_path=self.cfg.data.get('gwas', None),
                                      snp_count=self.cfg.experiment.get('snp_count', None),
                                      sample_indices=sample_index.train)
        X_val = load_from_pgen(self.cfg.data.genotype,
                                    gwas_path=self.cfg.data.get('gwas', None),
                                    snp_count=self.cfg.experiment.get('snp_count', None),
                                    sample_indices=sample_index.val)
        X_test = load_from_pgen(self.cfg.data.genotype,
                                     gwas_path=self.cfg.data.get('gwas', None),
                                     snp_count=self.cfg.experiment.get('snp_count', None),
                                     sample_indices=sample_index.test)
        return X(X_train, X_val, X_test)

    def load_covariates(self) -> X:
        test_samples_limit = self.cfg.experiment.get('test_samples_limit', None)
        X_train = load_covariates(self.cfg.data.covariates.train)
        X_val = load_covariates(self.cfg.data.covariates.val)
        X_test = load_covariates(self.cfg.data.covariates.test)[:test_samples_limit, :]
        if numpy.isnan(X_train).sum() > 0:
            raise ValueError(f'X_train has {numpy.isnan(X_train).sum()} nan values')
        if numpy.isnan(X_val).sum() > 0:
            raise ValueError(f'X_val has {numpy.isnan(X_val).sum()} nan values')
        if numpy.isnan(X_test).sum() > 0:
            raise ValueError(f'X_test has {numpy.isnan(X_test).sum()} nan values')
        return X(X_train, X_val, X_test)

    def _load_pcs(self) -> X:
        X_train = load_plink_pcs(path=self.cfg.data.x_reduced.train, order_as_in_file=self.cfg.data.phenotype.train).values
        X_val = load_plink_pcs(path=self.cfg.data.x_reduced.val, order_as_in_file=self.cfg.data.phenotype.val).values
        X_test = load_plink_pcs(path=self.cfg.data.x_reduced.test, order_as_in_file=self.cfg.data.phenotype.test).values
        return X(X_train, X_val, X_test)

    def _load_sample_indices(self) -> SampleIndex:
        self.logger.info("Loading sample indices")
        test_samples_limit = self.cfg.experiment.get('test_samples_limit', None)
        si_train = get_sample_indices(self.cfg.data.genotype,
                                                       self.cfg.data.phenotype.train)
        si_val = get_sample_indices(self.cfg.data.genotype,
                                                     self.cfg.data.phenotype.val)
        si_test = get_sample_indices(self.cfg.data.genotype,
                                                      self.cfg.data.phenotype.test,
                                                      indices_limit=test_samples_limit)
        return SampleIndex(si_train, si_val, si_test)

    def _sample_weights(self, pheno_file: str) -> numpy.ndarray:
        sw_frame = pd.read_table(pheno_file + '.sw')
        return sw_frame.sample_weight.values

    def load_sample_weights(self) -> Y:
        if self.cfg.study != 'ukb' or not self.cfg.get('sample_weights', False):
            # all samples will have equal weights during evaluation
            return Y(None, None, None)
        self.logger.info("Loading sample weights")
        train_sw = self._sample_weights(self.cfg.data.phenotype.train)
        val_sw = self._sample_weights(self.cfg.data.phenotype.val)
        test_sw = self._sample_weights(self.cfg.data.phenotype.test)
        unique = numpy.unique(val_sw)
        self.logger.info(f'we loaded {numpy.array2string(unique, precision=1, floatmode="fixed")} unique weights in val')
        return Y(train_sw, val_sw, test_sw)


def calculate_sample_weights(populations_frame: pd.DataFrame, pheno_frame: pd.DataFrame) -> numpy.ndarray:
    merged = pheno_frame.merge(populations_frame, how='inner', on='IID')
    populations = merged['node_index'].values
    unique, counts = numpy.unique(populations, return_counts=True)

    # populations contains values from [0, number of populations)
    sw = [populations.shape[0]/counts[p] for p in populations]

    return sw

def load_plink_pcs(path, order_as_in_file=None):
    """ Loads PLINK's eigenvector matrix (e.g. to be used as X for TG). If @order_as_in_file is not None,
     reorder rows of the matrix to match (IID-wise) rows of the file """
    df = pd.read_csv(path, sep='\t').rename(columns={'#IID': 'IID'}).set_index('IID').iloc[:, 2:]

    if order_as_in_file is not None:
        y = pd.read_csv(order_as_in_file, sep='\t').set_index('IID')
        assert len(df) == len(y)
        df = df.reindex(y.index)

    return df
