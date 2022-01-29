class PCA(object):
    @staticmethod
    def pca(input_prefix: str, pca_config: dict) -> None:
        """ Runs PCA via PLINK """
        pass
        "plink_command_that_has_pca_config_as_parameters"

    @staticmethod
    def pc_scatterplot(input_prefix: str, output_tag: str) -> None:
        """ Visualises eigenvector with scatterplot [matrix] """
        pass
        "plotly_graph".write_html("smth_based_on_output_tag.html")

    def run(self, input_prefix: str, pca_config: dict, output_tag: str):
        self.pca(input_prefix=input_prefix, pca_config=pca_config)
        self.pc_scatterplot(input_prefix=input_prefix, output_tag=output_tag)
