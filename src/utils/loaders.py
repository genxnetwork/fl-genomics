import pandas as pd


def load_plink_pcs(path, order_as_in_file=None):
    """ Loads PLINK's eigenvector matrix (e.g. to be used as X for TG). If @order_as_in_file is not None,
     reorder rows of the matrix to match (IID-wise) rows of the file """
    df = pd.read_csv(path, sep='\t').rename(columns={'#IID': 'IID'}).set_index('IID').filter(regex='^PC*')
    if order_as_in_file is not None:
        y = pd.read_csv(order_as_in_file, sep='\t').set_index('IID')
        assert len(df) == len(y)
        df = df.reindex(y.index)
    return df
