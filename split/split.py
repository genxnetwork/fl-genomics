from ukb_loader import UKBDataLoader
import pandas as pd
import numpy as np

split_dir = '/gpfs/gpfs0/ukb_data/processed_data/fml'

loader = UKBDataLoader(split_dir, 'split', '21000', ['31'])
train, val, test = loader.load_train(), loader.load_val(), loader.load_test()
df = pd.concat((train, val, test))
df.columns = ['sex', 'ethnic_background']

# Leave only those samples that passed population QC
pop_qc_ids = pd.read_csv('pop_qc_ids.csv', index_col='IID')
df = df.loc[df.index.intersection(pop_qc_ids.index)]

df.ethnic_background[df.ethnic_background.isna()] = 0.0
df.ethnic_background = df.ethnic_background.astype('int')

# Drop samples with missing/prefer not to answer ethnic background
df = df[~df.ethnic_background.isin([-1, -3, 0])]

df['FID'] = df.index
df['IID'] = df.index

split_map = {
    1001: 0,
    3001: 1,
    3002: 1,
    3003: 1,
    4001: 2,
    4002: 2,
    4003: 2,
    5: 3,
    1: 4,
    1002: 4,
    1003: 4,
    2001: 4,
    2:4,
    2002: 4,
    2003: 4,
    2004: 4,
    3004: 4,
    3: 4,
    4: 4,
    6: 4,    
}

df['split_code'] = df.ethnic_background.map(split_map)

data_root = "/gpfs/gpfs0/ukb_data/fml"

# British split
british_split = df.loc[df.split_code == 0]
np.random.seed(32)
british_split['split'] = np.random.choice(list(range(5)), size=british_split.shape[0], replace=True)

for i in range(5):
    british_split.loc[(british_split.split == i), ['FID', 'IID', 'sex']]\
        .to_csv(f'{data_root}/british_split/split_ids/{i}.csv', index=False, sep='\t')

# Ethnic split
np.random.seed(32)
holdout_idx = np.random.choice(df.index, size=df.shape[0]//5, replace=False)

np.random.seed(32)
holdout_idx = np.random.choice(df.index, size=df.shape[0]//5, replace=False)

df['split']= None

df.loc[holdout_idx, 'split'] = 5

for i in range(5):
    df.loc[(df.split_code == i) & (df.split != 5), 'split'] = i
    
for i in range(6):
    df.loc[(df.split == i), ['FID', 'IID', 'sex']]\
        .to_csv(f'{data_root}/ethnic_split/split_ids/{i}.csv', index=False, sep='\t')