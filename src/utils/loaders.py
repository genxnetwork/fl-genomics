import pandas as pd


def load_plink_pcs(path):
    """ Loads PLINK's eigenvector matrix (e.g. to be used as X for TG) """
    return pd.read_csv(path, sep='\t').rename(columns={'#IID': 'IID'}).set_index('IID').filter(regex='^PC*')
