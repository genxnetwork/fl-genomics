import sys
sys.path.append("..")
sys.path.append("../dimred/src")

from ukb_loader import UKBDataLoader
import pandas as pd
from os.path import join

from utils.split import Split
from config.path import ukb_loader_dir, data_root
from config.split_config import split_map, uniform_split_config, non_iid_split_name, uneven_split_shares_list


phenotype_name = "standing_height"
phenotype_code = 50

loader = UKBDataLoader(ukb_loader_dir, 'split', str(phenotype_code), ['31', '21003'])
df = pd.concat((loader.load_train(), loader.load_val(), loader.load_test()))

df.columns = ['sex', 'age', phenotype_name]

df['FID'] = df.index
df['IID'] = df.index


df = df[~df.loc[:, phenotype_name].isna()]

num_ethnic_nodes = max(list(split_map.values()))+1
ethnic_split = Split(join(data_root, non_iid_split_name), phenotype_name, num_ethnic_nodes)
uniform_split = Split(join(data_root, uniform_split_config['uniform_split_name']), phenotype_name,
                      uniform_split_config['n_nodes'])
uneven_split = Split(join(data_root, 'uneven_split'),
                     phenotype_name,
                     len(uneven_split_shares_list)+1)


for split in [ethnic_split, uniform_split, uneven_split]:
    for node_index in range(split.node_count):
        idx = pd.read_csv(join(split.node_ids_dir, f'{node_index}.csv'), sep='\t', index_col='IID')
        print(f"ID length {len(idx.index)}, intersection ID length {len(idx.index.intersection(df.index))}")
        df.loc[idx.index.intersection(df.index), ['FID', 'IID', 'sex', 'age', phenotype_name]]\
            .to_csv(join(split.cov_pheno_dir, f'{phenotype_name}_node_{node_index}.csv'),
                    sep='\t', index=False)