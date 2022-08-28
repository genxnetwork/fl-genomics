import pandas as pd
import plotly.express as px
from abc import abstractmethod

import sys
sys.path.append('..')

from configs.split_config import ethnic_background_name_map
from configs.split_config import TG_SUPERPOP_DICT
from preprocess.splitter import SplitBase
from preprocess.splitter_tg import SplitTGHeter
from utils.plink import run_plink


class PCABase(object):

    @staticmethod
    def pca(input_prefix: str, pca_config: dict, output_path: str, bin_file_type='--pfile') -> None:
        """ Runs PCA via PLINK """
        run_plink(args_list=[bin_file_type, input_prefix,
                             '--out', output_path],
                  args_dict=pca_config)

    @staticmethod
    @abstractmethod
    def pc_scatterplot():
        pass

    def run(self, input_prefix: str, pca_config: dict, output_path: str, scatter_plot_path=None, bin_file_type='--pfile'):
        self.pca(input_prefix=input_prefix, pca_config=pca_config, output_path=output_path, bin_file_type=bin_file_type)
        if scatter_plot_path is not None:
            self.pc_scatterplot(pca_path=output_path, scatter_plot_path=scatter_plot_path)


class PCAUKB(PCABase):

    @staticmethod
    def pc_scatterplot(pca_path: str, scatter_plot_path: str) -> None:
        """ Visualises eigenvector with scatterplot [matrix] """
        eigenvec = pd.read_table(f'{pca_path}.eigenvec')[['IID', 'PC1', 'PC2']]
        eigenvec.index = eigenvec.IID
        eb_df = SplitBase().get_ethnic_background()
        eigenvec['ethnic_background_name'] = eb_df.loc[eigenvec.IID, 'ethnic_background'].map(
            ethnic_background_name_map)  # TODO might not work properly
        px.scatter(eigenvec, x='PC1', y='PC2', color='ethnic_background_name').write_html(scatter_plot_path)

    def run(self, input_prefix: str, pca_config: dict, output_path: str, scatter_plot_path=None,
            bin_file_type='--pfile'):
        self.pca(input_prefix=input_prefix, pca_config=pca_config, output_path=output_path, bin_file_type=bin_file_type)
        if scatter_plot_path is not None:
            self.pc_scatterplot(pca_path=output_path, scatter_plot_path=scatter_plot_path)


class PCATG(PCABase):

    @staticmethod
    def pc_scatterplot(pca_path: str, scatter_plot_path: str) -> None:
        """ Visualises eigenvector with scatterplot [matrix] """
        eigenvec = pd.read_table(f'{pca_path}.eigenvec')[['IID', 'PC1', 'PC2']]
        tg_df = SplitTGHeter().get_target()
        eigenvec = pd.merge(eigenvec, tg_df, on='IID')
        eigenvec['ethnic_background_name'] = eigenvec['pop'].replace(TG_SUPERPOP_DICT)
        px.scatter(eigenvec, x='PC1', y='PC2', color='split').write_html(scatter_plot_path)

    def run(self, input_prefix: str, pca_config: dict, output_path: str, scatter_plot_path=None,
            bin_file_type='--pfile'):
        self.pca(input_prefix=input_prefix, pca_config=pca_config, output_path=output_path, bin_file_type=bin_file_type)
        if scatter_plot_path is not None:
            self.pc_scatterplot(pca_path=output_path, scatter_plot_path=scatter_plot_path)
