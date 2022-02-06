import pandas as pd
import plotly.express as px

import sys
sys.path.append('..')

from config.path import data_root
from config.split_config import ethnic_background_name_map
from preprocess.split import SplitBase
from utils.plink import run_plink

class PCA(object):
    def __init__(self):
        pass
    
    @staticmethod
    def pca(input_prefix: str, pca_config: dict) -> None:
        """ Runs PCA via PLINK """
        output_prefix = input_prefix.replace('genotypes', 'pca')
        run_plink(args_list=['--pfile', input_prefix,
                             '--out', output_prefix],
                  args_dict=pca_config)
        return output_prefix

    @staticmethod
    def pc_scatterplot(input_prefix: str) -> None:
        """ Visualises eigenvector with scatterplot [matrix] """
        eigenvec = pd.read_table(f'{input_prefix}.eigenvec')[['IID', 'PC1', 'PC2']]
        eigenvec.index = eigenvec.IID
        eb_df = SplitBase({}).get_ethnic_background()
        eigenvec['ethnic_background_name'] = eb_df.loc[eigenvec.IID, 'ethnic_background'].map(ethnic_background_name_map)
        output_prefix = input_prefix.replace('pca', 'figures')
        px.scatter(eigenvec, x='PC1', y='PC2', color='ethnic_background_name').write_html(f'{output_prefix}_pca.html')

    def run(self, input_prefix: str, pca_config: dict):
        pca_prefix = self.pca(input_prefix=input_prefix, pca_config=pca_config)
        self.pc_scatterplot(input_prefix=pca_prefix)
