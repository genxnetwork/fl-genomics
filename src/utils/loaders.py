import pandas as pd


def load_plink_pcs(path):
    """ Loads PLINK's eigenvector matrix (e.g. to be used as X for TG) """
    return pd.read_csv(path, sep='\t').drop(columns=['#FID']).set_index('IID')
