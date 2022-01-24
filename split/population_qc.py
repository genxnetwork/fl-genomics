from ukb_loader import UKBDataLoader
import pandas as pd

split_dir = '/gpfs/gpfs0/ukb_data/processed_data/fml'

# Filter by missingness
loader = UKBDataLoader(split_dir, 'split', '22005', ['31'])
train, val, test = loader.load_train(), loader.load_val(), loader.load_test()
df = pd.concat((train, val, test))
df = df[df['22005'] <= 0.015] # Missingness threshold per sample: 1.5%

outlier_fields = ['22019', '22027']
# Sex chromosome aneuploidy, Outliers for heterozygosity or missing rate 

outliers = []

for outlier_field in outlier_fields:
    loader = UKBDataLoader(split_dir, 'split', outlier_field, ['31'])
    train, val, test = loader.load_train(), loader.load_val(), loader.load_test()
    outliers.append(pd.concat((train, val, test)))

# Drop outlier indices
df = df.loc[df.index.difference(outliers[1].index.union(outliers[0].index))]


df.to_csv('pop_qc_ids.csv', index=True, index_label="IID", columns=[]) # Save indices that passed QC