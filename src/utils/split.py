from os import makedirs
from os.path import join


class Split:
    def __init__(self, root_dir: str, phenotype: str, node_count: int, fold_count: int = 10) -> None:
        """Creates necessary directory structure for dimred and encapsulates path manipulation for nodes and folds

        Args:
            root_dir (str): Directory where all work with this split will be done.
            phenotype (str): Name of phenotype.
            node_count (int): Number of nodes in split. 
            fold_count (int, optional): Number of cv folds. Defaults to 10.
        """        
        self.root_dir = root_dir
        self.phenotype = phenotype
        self.node_count = node_count
        self.fold_count = fold_count

        self.cov_pheno_dir = join(self.root_dir, 'phenotypes')
        self.pca_dir = join(self.root_dir, 'pca')
        self.pca_cov_dir = join(self.root_dir, 'covariates')
        self.phenotypes_dir = join(self.root_dir, 'only_phenotypes')
        self.node_ids_dir = join(self.root_dir, 'split_ids')
        self.gwas_dir = join(self.root_dir, 'gwas')
        self.snplists_dir = join(self.root_dir, 'gwas', 'snplists')
        self.genotypes_dir = join(self.root_dir, 'genotypes')

        self._create_dir_structure()

    def _create_dir_structure(self):
        for _dir in [self.cov_pheno_dir,
                     self.pca_dir,
                     self.pca_cov_dir,
                     self.phenotypes_dir,
                     self.node_ids_dir,
                     self.gwas_dir,
                     self.snplists_dir,
                     self.genotypes_dir] + \
                     [join(self.genotypes_dir, f"node_{node_index}") \
                         for node_index in range(self.node_count)] + \
                     [join(self.pca_dir, f"node_{node_index}") \
                         for node_index in range(self.node_count)]:
            makedirs(_dir, exist_ok=True)

        makedirs(join(self.snplists_dir, self.phenotype), exist_ok=True)

        for node_index in range(self.node_count):
            makedirs(join(self.node_ids_dir, f'node_{node_index}'), exist_ok=True)
            makedirs(join(self.cov_pheno_dir, self.phenotype, f'node_{node_index}'), exist_ok=True)
            makedirs(join(self.phenotypes_dir, self.phenotype, f'node_{node_index}'), exist_ok=True)
            makedirs(join(self.pca_dir, f'node_{node_index}'), exist_ok=True)
            makedirs(join(self.pca_cov_dir, self.phenotype, f'node_{node_index}'), exist_ok=True)
            makedirs(join(self.gwas_dir, self.phenotype, f'node_{node_index}'), exist_ok=True)

    def get_source_ids_path(self, node_index: int) -> str:
        return join(self.node_ids_dir, f'{node_index}.csv')

    def get_ids_path(self, node_index: int, fold_index: int, part_name: str) -> str:
        return join(self.node_ids_dir, f'node_{node_index}', f'fold_{fold_index}_{part_name}.tsv')

    def get_source_phenotype_path(self, node_index: int) -> str:
        # TODO: rename split to node in preprocessing
        return join(self.cov_pheno_dir, f'{self.phenotype}_split_{node_index}.csv')

    def get_cov_pheno_path(self, node_index: int, fold_index: int, part: str) -> str:
        return join(self.cov_pheno_dir, self.phenotype, f'node_{node_index}', f'fold_{fold_index}_{part}.tsv')

    def get_phenotype_path(self, node_index: int, fold_index: int, part: str, adjusted: bool = False) -> str:
        if adjusted:
            return join(self.phenotypes_dir, self.phenotype, f'node_{node_index}', f'fold_{fold_index}_{part}.adjusted.tsv')
        else:
            return join(self.phenotypes_dir, self.phenotype, f'node_{node_index}', f'fold_{fold_index}_{part}.tsv')

    def get_source_pca_path(self, node_index: int) -> str:
        return join(self.pca_dir, f'{node_index}_projections.csv.eigenvec')

    def get_pca_path(self, node_index: int, fold_index: int, part: str, ext: str = '.csv.eigenvec') -> str:            
        return join(self.pca_dir, f'node_{node_index}', f'fold_{fold_index}_{part}_projections') + ext
    
    def get_pca_cov_path(self, node_index: int, fold_index: int, part: str) -> str:
        return join(self.pca_cov_dir, self.phenotype, f'node_{node_index}', f'fold_{fold_index}_{part}.tsv')

    def get_gwas_path(self, node_index: int, fold_index: int, adjusted: bool = False) -> str:
        return join(self.gwas_dir, self.phenotype, f'node_{node_index}', f'fold_{fold_index}.adjusted.tsv' if adjusted else f'fold_{fold_index}.tsv')

    def get_snplist_path(self, strategy: str, node_index: int, fold_index: int) -> str:
        return join(self.snplists_dir, self.phenotype, f'node_{node_index}_fold_{fold_index}_{strategy}.snplist')

    def get_source_pfile_path(self, node_index: int) -> str:
        return join(self.genotypes_dir, f'node_{node_index}_filtered')

    def get_pfile_path(self, node_index: int, fold_index: int, part_name:str):
        return join(self.genotypes_dir, f"node_{node_index}", f"fold_{fold_index}_{part_name}")
    
    def get_topk_pfile_path(self, strategy: str, node_index: int, fold_index: int, snp_count: int, part: str, sample_count: int = None) -> str:
        fold_dir =  join(self.genotypes_dir, self.phenotype, f'node_{node_index}', strategy, f'fold_{fold_index}')
        makedirs(fold_dir, exist_ok=True)
        if sample_count is None:
            return join(fold_dir, f'top_{snp_count}_{part}')
        else:
            return join(fold_dir, f'top_{snp_count}_{part}_samples_{sample_count}')

