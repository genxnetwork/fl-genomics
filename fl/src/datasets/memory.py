import numpy
import pandas
from pgenlib import PgenReader
from sklearn.linear_model import LinearRegression


def load_from_pgen(pfile_path: str, gwas_path: str, snp_count: int, missing='zero') -> numpy.ndarray:
    """
    Loads genotypes from .pgen into numpy array and selects top {snp_count} snps

    Args:
        pfile_path (str): Path to plink 2.0 .pgen, .pvar, .psam dataset. It should not have a .pgen extension.
        gwas_path (str): Path to plink 2.0 GWAS results file generated by plink 2.0 --glm. 
        snp_count (int): Number of most significant SNPs to load. If None then load all SNPs
        missing (str): Strategy of filling missing values. Default is 'zero', i.e. homozygous reference value. Other is 'mean'.

    Raises:
        ValueError: If snp_count is greated than number of SNPs in .pgen

    Returns:
        numpy.ndarray: An int8 sample-major array with {snp_count} genotypes
    """    
    reader = PgenReader((pfile_path + '.pgen').encode('utf-8'))
    max_snp_count = reader.get_variant_ct()
    sample_count = reader.get_raw_sample_ct()
    if snp_count is not None and snp_count > max_snp_count:
        raise ValueError(f'snp_count {snp_count} should be not greater than max_snp_count {max_snp_count}')
    
    snp_count = max_snp_count if snp_count is None else snp_count
    array = numpy.empty((sample_count, snp_count), dtype=numpy.int8)
    
    if snp_count is None or snp_count == max_snp_count:
        reader.read_range(0, max_snp_count, array, sample_maj=True)
    else:
        snp_indices = get_snp_list(pfile_path, gwas_path, snp_count)
        reader.read_list(snp_indices, array, sample_maj=True)
    if missing == 'zero':
        array[array < 0] = 0
    elif missing == 'mean':
        array = numpy.where(numpy.isnan(array), numpy.nanmean(array, axis=0), array) 
    return array


def load_phenotype(phenotype_path: str) -> numpy.ndarray:
    data = pandas.read_table(phenotype_path)
    return data.iloc[:, -1].values

def load_covariates(covariates_path: str) -> numpy.ndarray:
    data = pandas.read_table(covariates_path)
    return data.iloc[:, 2:].values


def get_snp_list(pfile_path: str, gwas_path: str, snp_count: int) -> numpy.ndarray:
    pvar = pandas.read_table(pfile_path + '.pvar')
    gwas = pandas.read_table(gwas_path)
    gwas.sort_values(by='LOG10_P', axis='index', ascending=False, inplace=True)
    snp_ids = set(gwas.ID.values[:snp_count])
    snp_indices = numpy.arange(pvar.shape[0])[pvar.ID.isin(snp_ids)].astype(numpy.uint32)
    return snp_indices

def get_phenotype_adjuster(phenotype_path_tr: str, covariates_path_tr: str) -> LinearRegression:
    """
    Returns a LinearRegression model used to adjust phenotypes to control for all of the given covariates.
    Expects a file with only the phenotype.
    """
    X_tr = load_covariates(covariates_path_tr)
    y_tr = load_phenotype(phenotype_path_tr)
    
    return LinearRegression().fit(X_tr, y_tr)

