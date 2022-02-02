from ukb_loader import UKBDataLoader
import pandas as pd

def central_qc(split_dir: str, valid_ids_path: str) -> None:
    # Filter by missingness
    loader = UKBDataLoader(split_dir, 'split', '22005', ['31'])
    df = pd.concat((loader.load_train(), loader.load_val(), loader.load_test()))
    df = df[df['22005'] <= 0.015] # Missingness threshold per sample: 1.5%

    # Sex chromosome aneuploidy, Outliers for heterozygosity or missing rate 
    outlier_fields = ['22019', '22027']

    outliers = []

    for outlier_field in outlier_fields:
        loader = UKBDataLoader(split_dir, 'split', outlier_field, ['31'])
        outliers.append(pd.concat((loader.load_train(), loader.load_val(), loader.load_test())))

    # Drop outlier indices
    df = df.loc[df.index.difference(outliers[1].index.union(outliers[0].index))]

    df.to_csv(valid_ids_path, index=True, index_label="IID", columns=[])
