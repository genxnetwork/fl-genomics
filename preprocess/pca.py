import sys
sys.path.append('../utils')

from config.path import data_root
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
    def pc_scatterplot(input_prefix: str, output_tag: str) -> None:
        """ Visualises eigenvector with scatterplot [matrix] """
        pass
        # TODO
        #"plotly_graph".write_html("smth_based_on_output_tag.html")

    def run(self, input_prefix: str, pca_config: dict, output_tag: str):
        self.pca(input_prefix=input_prefix, pca_config=pca_config, output_tag=output_tag)
        # TODO
        #self.pc_scatterplot(input_prefix=input_prefix, output_tag=output_tag)
