from utils.plink import run_plink
import sys
from os import path
sys.path.append('dimred/src')
from utils.split import Split
from config.path import data_root
from config.split_config import heterogeneous_split_name, n_heterogeneous_nodes
from preprocess.split import SplitHeterogeneous

heterogeneous_split = Split(path.join(data_root, heterogeneous_split_name), 'standing_height', n_heterogeneous_nodes+1)
heterogeneous_splitter = SplitHeterogeneous()
print("Saving IDs")
heterogeneous_splitter.save_all_ids()
print("Making pgen")
heterogeneous_splitter.make_split_pgen(heterogeneous_split.get_source_ids_path(node_index=0),
                                       path.join(heterogeneous_split.genotypes_dir, 'node_0'))
print("Running PCA")
run_plink(args_list=['--pfile', path.join(heterogeneous_split.genotypes_dir, 'node_0'),
                     '--freq', 'counts',
                     '--out', path.join(heterogeneous_split.pca_dir, 'node_0'),
                     '--pca', 'allele-wts', '20', 'approx', '--threads', '8'],
# )
print("Saving cluster_ids")
heterogeneous_splitter.split(pca_path=path.join(heterogeneous_split.pca_dir, 'node_0.eigenvec'))
