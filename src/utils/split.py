import os
from os import makedirs


class Split:
    def __init__(self, root_dir: str, phenotype: str, node_count: int = None, nodes: list = None, fold_count: int = 10) -> None:
        """Creates necessary directory structure for dimred and encapsulates path manipulation for nodes and folds

        Args:
            root_dir (str): Directory where all work with this split will be done.
            phenotype (str): Name of phenotype.
            node_count (int): Number of nodes in split. 
            nodes (list): Alternatively, explicit node names.
            fold_count (int, optional): Number of cv folds. Defaults to 10.
        """        
        self.root_dir = root_dir
        self.phenotype = phenotype
        self.node_count = node_count
        self.nodes = nodes
        self.fold_count = fold_count

        self.cov_pheno_dir = os.path.join(self.root_dir, 'phenotypes')
        self.pca_dir = os.path.join(self.root_dir, 'pca')
        self.pca_cov_dir = os.path.join(self.root_dir, 'covariates')
        self.phenotypes_dir = os.path.join(self.root_dir, 'only_phenotypes')
        self.node_ids_dir = os.path.join(self.root_dir, 'split_ids')
        self.gwas_dir = os.path.join(self.root_dir, 'gwas')
        self.snplists_dir = os.path.join(self.root_dir, 'gwas', 'snplists')
        self.genotypes_dir = os.path.join(self.root_dir, 'genotypes')

        self._create_dir_structure()

    def _create_dir_structure(self):
        nodes = [f'node_{i}' for i in range(self.node_count)] if self.nodes is None else self.nodes  # assign node names
        for _dir in [self.cov_pheno_dir,
                     self.pca_dir,
                     self.pca_cov_dir,
                     self.phenotypes_dir,
                     self.node_ids_dir,
                     self.gwas_dir,
                     self.snplists_dir,
                     self.genotypes_dir] + \
                     [os.path.join(self.genotypes_dir, node) for node in nodes] + \
                     [os.path.join(self.pca_dir, node) for node in nodes]:
            makedirs(_dir, exist_ok=True)

        makedirs(os.path.join(self.snplists_dir, self.phenotype), exist_ok=True)

        for node in nodes:
            makedirs(os.path.join(self.node_ids_dir, node), exist_ok=True)
            makedirs(os.path.join(self.cov_pheno_dir, self.phenotype, node), exist_ok=True)
            makedirs(os.path.join(self.phenotypes_dir, self.phenotype, node), exist_ok=True)
            makedirs(os.path.join(self.pca_dir, node), exist_ok=True)
            makedirs(os.path.join(self.pca_cov_dir, self.phenotype, node), exist_ok=True)
            makedirs(os.path.join(self.gwas_dir, self.phenotype, node), exist_ok=True)

    def get_source_ids_path(self, fn: str) -> str:
        return os.path.join(self.node_ids_dir, fn)

    def get_ids_path(self, fold_index: int, part_name: str, node_index: int = None, node: str = None) -> str:
        if node_index is not None:
            return os.path.join(self.node_ids_dir, f'node_{node_index}', f'fold_{fold_index}_{part_name}.tsv')
        else:
            return os.path.join(self.node_ids_dir, node, f'fold_{fold_index}_{part_name}.tsv')

    def get_source_phenotype_path(self, node_index: int) -> str:
        # TODO: rename split to node in preprocessing
        return os.path.join(self.cov_pheno_dir, f'{self.phenotype}_split_{node_index}.csv')

    def get_cov_pheno_path(self, node_index: int, fold_index: int, part: str) -> str:
        return os.path.join(self.cov_pheno_dir, self.phenotype, f'node_{node_index}', f'fold_{fold_index}_{part}.tsv')

    def get_phenotype_path(self, node_index: int, fold_index: int, part: str, adjusted: bool = False) -> str:
        if adjusted:
            return os.path.join(self.phenotypes_dir, self.phenotype, f'node_{node_index}', f'fold_{fold_index}_{part}.adjusted.tsv')
        else:
            return os.path.join(self.phenotypes_dir, self.phenotype, f'node_{node_index}', f'fold_{fold_index}_{part}.tsv')

    def get_source_pca_path(self, node_index: int = None, node: str = None) -> str:
        return os.path.join(self.pca_dir, f'{node_index}_projections.csv.eigenvec' if node_index is not None else f'{node}_projections.csv.eigenvec')

    def get_pca_path(self, fold_index: int, part: str, node_index: int = None, node: str = None, ext: str = '.csv.eigenvec') -> str:
        return os.path.join(self.pca_dir, f'node_{node_index}' if node_index is not None else node, f'fold_{fold_index}_{part}_projections') + ext
    
    def get_pca_cov_path(self, fold_index: int, part: str, node_index: int = None, node: str = None) -> str:
        return os.path.join(self.pca_cov_dir, self.phenotype, f'node_{node_index}' if node_index is not None else node, f'fold_{fold_index}_{part}.tsv')

    def get_gwas_path(self, node_index: int, fold_index: int, adjusted: bool = False) -> str:
        return os.path.join(self.gwas_dir, self.phenotype, f'node_{node_index}', f'fold_{fold_index}.adjusted.tsv' if adjusted else f'fold_{fold_index}.tsv')

    def get_snplist_path(self, strategy: str, node_index: int, fold_index: int) -> str:
        return os.path.join(self.snplists_dir, self.phenotype, f'node_{node_index}_fold_{fold_index}_{strategy}.snplist')

    def get_source_pfile_path(self, node_index: int = None, node: str = None) -> str:
        return os.path.join(self.genotypes_dir, f'node_{node_index}_filtered' if node_index is not None else f'{node}_filtered')

    def get_pfile_path(self, fold_index: int, part_name: str, node_index: int = None, node: str = None):
        return os.path.join(self.genotypes_dir, f"node_{node_index}" if node_index is not None else node, f"fold_{fold_index}_{part_name}")
    
    def get_topk_pfile_path(self, strategy: str, node_index: int, fold_index: int, snp_count: int, part: str, sample_count: int = None) -> str:
        fold_dir = os.path.join(self.genotypes_dir, self.phenotype, f'node_{node_index}', strategy, f'fold_{fold_index}')
        makedirs(fold_dir, exist_ok=True)
        if sample_count is None:
            return os.path.join(fold_dir, f'top_{snp_count}_{part}')
        else:
            return os.path.join(fold_dir, f'top_{snp_count}_{part}_samples_{sample_count}')

