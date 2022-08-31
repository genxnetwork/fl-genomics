import logging
import os

import dash_bio
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s')
DATA_DIR = '/mnt/genx-bio-share/UKB/gwas_analysis'
out_dir_root = os.path.join(DATA_DIR, 'out')
os.makedirs(out_dir_root, exist_ok=True)

class GwasWbSubsample(object):
    def __init__(self, pheno_name, num_repeats=10, maf_thresholds=[0.01, 0.02, 0.05, 0.1], top_snps_list=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 596783]):
        self.folder = os.path.join(DATA_DIR, pheno_name)
        self.num_repeats = num_repeats
        self.maf_thresholds = maf_thresholds
        self.top_snps_list = top_snps_list
        self.pheno_name = pheno_name
        logging.info('Reading data...')
        maf_df = pd.read_csv(os.path.join(DATA_DIR, 'plink2.afreq'), sep='\t', usecols=['ID', 'ALT_FREQS'], dtype={'ID': object, 'ALT_FREQS': float}).set_index('ID')
        all_df = pd.read_csv(os.path.join(self.folder, f'{pheno_name}_all_samples.tsv'), sep='\t', na_values='.',
                             usecols=['ID', 'LOG10_P', 'BETA', 'ERRCODE'],
                             dtype={'ID': object, 'LOG10_P': float, 'BETA': float, 'ERRCODE': object}).set_index('ID')
        pd.testing.assert_index_equal(maf_df.index, all_df.index)
        sub_df_list = [pd.read_csv(os.path.join(self.folder, f'{pheno_name}_fold_{i}.{pheno_name}.glm.linear'), sep='\t',
                                   na_values='.', usecols=['ID', 'LOG10_P', 'BETA'],
                                   dtype={'ID': object, 'LOG10_P': float, 'BETA': float})
                       .rename(columns={'LOG10_P': f'LOG10_P_{i}', 'BETA': f'BETA_{i}'}).set_index('ID') for i in range(1, num_repeats)]
        [pd.testing.assert_index_equal(maf_df.index, dfi.index) for dfi in sub_df_list]
        logging.info('Joining data...')
        self.df = pd.concat([maf_df, all_df, *sub_df_list], axis=1)
        logging.info(f'Current shape {self.df.shape}...')
        correct_ids = self.df['ERRCODE'].isna()
        logging.info(f'Leaving {correct_ids.sum()} correct SNPs and dropping the ERRCODE column...')
        self.df = self.df[correct_ids].drop(columns='ERRCODE')
        logging.info(f'Current shape {self.df.shape}...')
        logging.info(f'Sorting by LOG10_P...')
        self.df = self.df.sort_values('LOG10_P', ascending=False)
        self.out_dir = os.path.join(self.folder, 'out')
        os.makedirs(self.out_dir, exist_ok=True)
        # self.df.to_csv(os.path.join(self.out_dir, 'df.csv'))
        # self.df = pd.read_csv(os.path.join(self.out_dir, 'df.csv')).set_index('ID')

    def marginal_distributions(self):
        logging.info('One-dimenstional stats...')
        # cols = ['ALT_FREQS', 'LOG10_P'] + [f'LOG10_P_{i}' for i in range(self.num_repeats)]
        self.df.describe().to_csv(os.path.join(self.out_dir, f'one_dim.csv'))
        self.df.describe().to_csv(os.path.join(out_dir_root, f'{self.pheno_name}_one_dim.csv'))
        # for col in self.df.columns:
        #     logging.info(f'Histogram for column {col}...')
        #     px.histogram(self.df, x=col).write_html(os.path.join(self.out_dir, f'hist_{col}.html'))

    def make_figure(self, df):
        logging.info(f'Plotting joint distribution...')
        df = df.reset_index()
        df['maf'] = df['maf'].astype(str) + '_'
        df['error_y'] = df['0.75'] - df['0.5']
        df['error_y_minus'] = df['0.5'] - df['0.25']
        fig = px.scatter(df, x="top_snps", y="0.5", color="maf",
                   error_y="error_y", error_y_minus="error_y_minus", log_x=True)
        fig.add_trace(go.Line(x=self.top_snps_list, y=[x / max(self.top_snps_list) for x in self.top_snps_list]))
        fig.write_html(os.path.join(self.out_dir, 'joint_dist.html'))
        fig.write_html(os.path.join(out_dir_root, f'{self.pheno_name}_joint_dist.html'))
        pass

    def joint_distribution(self):
        logging.info(f'Analysing joint distribution...')
        subs_cols = [f'LOG10_P_{i}' for i in range(1, self.num_repeats)]
        values_df = pd.DataFrame(index=self.maf_thresholds, columns=self.top_snps_list)
        for maf in self.maf_thresholds:
            sdf = self.df[self.df['ALT_FREQS'] >= maf]
            # snp_order = pd.DataFrame(
            #     {col: sdf[col].sample(frac=1).index.to_list() for col in subs_cols},
            #     index=sdf.index)
            snp_order = pd.DataFrame(
                {col: sdf[col].sort_values(ascending=False).index.to_list() for col in subs_cols},
                index=sdf.index)
            for ts in self.top_snps_list:
                tsdf = snp_order.head(ts)
                values_df.loc[maf, ts] = [len(set(tsdf.index).intersection(tsdf[col])) / len(tsdf) for col in subs_cols]
        ser = values_df.stack()
        values_df = pd.DataFrame.from_dict(dict(zip(ser.index, ser.values))).T
        values_df.index.names = ['maf', 'top_snps']
        quantiles_df = values_df.quantile([0.25, 0.5, 0.75], axis=1).T
        quantiles_df.columns = quantiles_df.columns.astype(str)
        self.make_figure(quantiles_df)


    def analyse(self):
        self.get_manh()
        self.joint_distribution()
        self.marginal_distributions()
        pass

    def get_manh(self):
        subs_cols = [f'LOG10_P_{i}' for i in range(1, self.num_repeats)]
        maf = 0.05
        ts = 1000
        col = subs_cols[5]
        sdf = self.df[self.df['ALT_FREQS'] >= maf]
        snp_order = pd.DataFrame(
            {col: sdf[col].sort_values(ascending=False).index.to_list() for col in subs_cols},
            index=sdf.index)
        tsdf = snp_order.head(ts)
        tmp = set(tsdf.index).intersection(tsdf[col])
        # tmp1 = self.df.loc[list(tmp), 'ID']
        tmp_df = pd.read_csv(os.path.join(self.folder, f'{self.pheno_name}_all_samples.tsv'), sep='\t')
        tmp_df = tmp_df[tmp_df['ID'].isin(tmp)]
        tmp_df = tmp_df[tmp_df['#CHROM'].astype(str).str.isnumeric()]
        tmp_df['#CHROM'] = tmp_df['#CHROM'].astype(int)
        # tmp_df['P'] = [10 ** (-x) for x in tmp_df['LOG10_P']]
        fig_manh = dash_bio.ManhattanPlot(
            dataframe=tmp_df,
            chrm='#CHROM', bp='POS', p='LOG10_P', snp='ID',
            gene=None,
            highlight_color='#00FFAA',
            suggestiveline_color='#AA00AA',
            genomewideline_color='#AA5500'
        )
        fig_manh.write_html(os.path.join(self.out_dir, f'manh.html'))
        fig_manh.write_html(os.path.join(out_dir_root, f'{self.pheno_name}_manh.html'))


if __name__ == '__main__':
    for pheno_name in [
        'standing_height',
        'basal_metabolic_rate',
 'body_mass_index',
 'erythrocyte_count',
 'forced_vital_capacity',
 'hls_reticulocyte_count',
 'platelet_count',
 'platelet_volume',
 'reticulocyte_count',
 ]:
        logging.info(f'Processing phenotype {pheno_name}')
        gws = GwasWbSubsample(pheno_name=pheno_name)
        gws.analyse()
