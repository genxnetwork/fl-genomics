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
    def pca(input_prefix: str, pca_config: dict, output_tag: str) -> None:
        """ Runs PCA via PLINK """
        run_plink(args_list=['--pfile', input_prefix,
                             '--out', f"{data_root}/pca/{output_tag}"],
                  args_dict=pca_config)

    @staticmethod
    def pc_scatterplot(tag: str) -> None:
        """ Visualises eigenvector with scatterplot [matrix] """
        eigenvec = pd.read_table(f'{data_root}/pca/{tag}.eigenvec')[['IID', 'PC1', 'PC2']]
        eigenvec.index = eigenvec.IID
        eb_df = SplitBase({}).get_ethnic_background()
        eigenvec['ethnic_background_name'] = eb_df.loc[eigenvec.IID, 'ethnic_background'].map(ethnic_background_name_map)
        px.scatter(eigenvec, x='PC1', y='PC2', color='ethnic_background_name').write_html(f'{data_root}/figures/{tag}_pca.html')

    def run(self, input_prefix: str, pca_config: dict, output_tag: str):
        self.pca(input_prefix=input_prefix, pca_config=pca_config, output_tag=output_tag)
        self.pc_scatterplot(tag=output_tag)
