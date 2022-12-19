import os
import pandas as pd

import mlflow
import plotly.express as px

from mlflow.tracking import MlflowClient

import sys
sys.path.append('..')

from utils.mlflow import mlflow_get_results_table, folds_to_quantiles

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

SELECTED_TAGS = ['tags.phenotype', 'tags.model', 'tags.dataset', 'tags.snp_count', 'tags.sample_count',
                 'tags.fold_index']
SELECTED_METRICS = ['metrics.test_r2', 'metrics.train_r2']

QUANTITATIVE_PHENOTYPES = ['alkaline_phosphatase',
                           'apolipoprotein_a', 
                           'cystatin_c',
                           'erythrocyte_count',
                           'hdl_cholesterol',
                           'platelet_volume',
                           'shbg',
                           'standing_height',
                           'triglycerides'
                           ]
BINARY_PHENOTYPES = ['asthma', 'diabetes', 'hypothyroidism', 'hypertension', 'psoriasis', 'rhinitis']

snps='local'

EXPERIMENT_NAME_DICT = {
    'continuous': {
        'local': {
            'local_local': 'local-models-local-snps',
            'local_global': 'local-models-local-snps-centralized-metrics',
            'centr': 'centralized-models-centralized-snps-fixed',
            },
        'meta': {
            'local_local': 'local-models-meta-snps',
            'local_global': 'local-models-meta-snps-centralized-metrics',
            'centr': 'centralized-models-meta-snps-fixed',
            },
        'federated': 'fl-ukb-mg-avg',
        'cov_only': 'covariates-only',

        },
    'binary': {
        'local': {
            'local_local': 'categorical-local-models-local-snps',
            'local_global': 'categorical-local-models-local-snps-centralized-metrics',
            'centr': 'categorical-centralized-models-centralized-snps',
            },
        'meta': {
            'local_local': 'categorical-local-models-meta-snps',
            'local_global': 'categorical-local-models-meta-snps-centralized-metrics',
            'centr': 'categorical-centralized-models-meta-snps',
            },
        'federated': 'fl-ukb-mg-avg-bin',
        'cov_only': 'categorical-covariates-only',
        }
}

def filter_pheno(df, pheno_list=QUANTITATIVE_PHENOTYPES+BINARY_PHENOTYPES):
    return df[df['tags.phenotype'].isin(pheno_list)]

def ethnicity_from_dataset(df: pd.DataFrame):
    df['tags.ethnicity'] = df['tags.dataset'].str.split('_').str[0]
    return df


def normalize_by_centr_median(df: pd.DataFrame, mode=None):
    if mode == 'centralized_minus_covariates':
        df.loc[:, ['lower', 'median', 'upper']] = (df.loc[:, ['lower', 'median', 'upper']] -
                                                   df.loc[(df['type']=='covariates_only') & (df['tags.dataset'].str.contains('Centralized')), 'median'].squeeze()) / \
                                                   (df.loc[df['type']=='centralized', 'median'].squeeze() - df.loc[(df['type']=='covariates_only') & (df['tags.dataset'].str.contains('Centralized')), 'median'].squeeze())

    elif mode == 'centralized':
        df.loc[:, ['lower', 'median', 'upper']] = df.loc[:, ['lower', 'median', 'upper']] / df.loc[df['type']=='centralized', 'median'].squeeze()

    return df


def read_local_local(exp_name, selected_metrics, exp_type='local-localtestset'):
    df_local_localtestset = mlflow_get_results_table(client=MlflowClient(), experiment_name=exp_name,
                                                     selected_tags=SELECTED_TAGS + ['tags.node_index'], selected_metric=selected_metrics,
                                                     cols_to_int=['tags.node_index', 'tags.snp_count', 'tags.sample_count'],
                                                     custom_transform=filter_pheno)
    df_local_localtestset['type'] = exp_type
    return df_local_localtestset

def read_covariates_only(exp_name, selected_metrics):
    df_covariates_only = mlflow_get_results_table(client=MlflowClient(), experiment_name=exp_name,
                                                     selected_tags=SELECTED_TAGS + ['tags.node_index'], selected_metric=selected_metrics,
                                                     cols_to_int=['tags.node_index', 'tags.snp_count', 'tags.sample_count'],
                                                     custom_transform=filter_pheno)
    df_covariates_only['type'] = 'covariates_only'
    return df_covariates_only


def read_local_global(exp_name, selected_metrics, exp_type='local-globaltestset'):
    df_local_globaltestset = mlflow_get_results_table(client=MlflowClient(), experiment_name=exp_name,
                                  selected_tags=['tags.phenotype', 'tags.fold_index', 'tags.train_node_index'], selected_metric=selected_metrics,
                                  cols_to_int=['tags.train_node_index'],
                                  custom_transform=filter_pheno)
    df_local_globaltestset['type'] = exp_type
    return df_local_globaltestset


def read_centr(exp_name, selected_metrics, exp_type='centralized'):
    df_centr = mlflow_get_results_table(client=MlflowClient(), experiment_name=exp_name,
                                  selected_tags=SELECTED_TAGS, selected_metric=selected_metrics,
                                  cols_to_int=['tags.snp_count', 'tags.sample_count'],
                                  custom_transform=filter_pheno)
    df_centr['type'] = exp_type
    return df_centr


def read_feder(exp_name, selected_metrics):
    df_feder = mlflow_get_results_table(client=MlflowClient(), experiment_name=exp_name,
                                        selected_tags=['tags.phenotype', 'tags.fold'],
                                        selected_metric=selected_metrics,
                                        dropna_col=['tags.fold'],
                                         custom_transform=filter_pheno,
                                        cols_to_int=['tags.fold'])\
        .rename(columns={'tags.fold': 'tags.fold_index'})
    df_feder['type'] = 'federated'
    return df_feder


def impute_missing_columns(df_local_globaltestset, df_meta_globaltestset, df_feder, df_local_localtestset, df_centr, df_cov):
    # dfs to impute missing tags for local-global
    node_index_df = df_local_localtestset[['tags.node_index', 'tags.dataset']].drop_duplicates()
    samcount_index_df = df_local_localtestset[['tags.node_index', 'tags.sample_count']].drop_duplicates()

    df_local_globaltestset['tags.model'] = 'lassonet'
    df_local_globaltestset['tags.snp_count'] = df_local_localtestset['tags.snp_count'].iloc[0]
    df_local_globaltestset['tags.sample_count'] = df_local_globaltestset['tags.train_node_index'].replace(dict(zip(samcount_index_df['tags.node_index'], samcount_index_df['tags.sample_count'])))
    df_local_globaltestset['tags.dataset'] = df_local_globaltestset['tags.train_node_index'].replace(dict(zip(node_index_df['tags.node_index'], node_index_df['tags.dataset'])))

    df_meta_globaltestset['tags.model'] = 'lassonet'
    df_meta_globaltestset['tags.snp_count'] = df_local_localtestset['tags.snp_count'].iloc[0]
    df_meta_globaltestset['tags.sample_count'] = df_meta_globaltestset['tags.train_node_index'].replace(dict(zip(samcount_index_df['tags.node_index'], samcount_index_df['tags.sample_count'])))
    df_meta_globaltestset['tags.dataset'] = df_meta_globaltestset['tags.train_node_index'].replace(dict(zip(node_index_df['tags.node_index'], node_index_df['tags.dataset'])))

    df_feder['tags.model'] = 'lassonet'
    df_feder['tags.snp_count'] = df_local_localtestset['tags.snp_count'].iloc[0]
    df_feder['tags.sample_count'] = df_centr['tags.sample_count']
    df_feder['tags.dataset'] = 'federated'

    # df_cov['tags.dataset'] = 'covariates_only'
    df_cov['tags.model'] = 'lassonet'
    df_cov['tags.snp_count'] = df_local_localtestset['tags.snp_count'].iloc[0]
    # df_cov['tags.sample_count'] = df_centr['tags.sample_count']
    return df_local_globaltestset, df_meta_globaltestset, df_feder, df_cov


def prepare_data_to_plot(df_local_localtestset, df_meta_localtestset,
                         df_local_globaltestset, df_meta_globaltestset,
                         df_centr, df_centr_meta,
                         df_feder, df_cov, metric_name):
    # concat datasets
    df = pd.concat([
                    # df_local_localtestset.drop(columns=['tags.node_index']),
                    df_local_globaltestset.drop(columns=['tags.train_node_index']),
                    df_centr,
                    # df_meta_localtestset.drop(columns=['tags.node_index']),
                    df_meta_globaltestset.drop(columns=['tags.train_node_index']),
                    # df_centr_meta,
                    df_feder,
                    df_cov.drop(columns=['tags.node_index'])
                    ])

    # converting folds to quantiles (0.1, median, 0.9)
    df = folds_to_quantiles(df.query("metric_name == @metric_name"), fold_col='tags.fold_index', val_col='metric_value')
    # for each phenotype normalize all quantiles and types (local, centralized, federated) by centralized median

    df = df.set_index(['tags.phenotype']).groupby(['tags.phenotype'])[['type', 'tags.dataset', 'tags.sample_count', 'lower', 'median',
       'upper']].apply(normalize_by_centr_median).reset_index()
    return df


def plot(df, df_local_globaltestset, out_fn):
    # plot (still not in its final form yet)
    df['error_up'] = df['upper'] - df['median']
    df['error_down'] = df['median'] - df['lower']
    df.columns = [i.replace('tags.', '') for i in df.columns]
    df['phenotype'] = df['phenotype'].replace({'alkaline_phosphatase': 'Alkaline phosphatase', 'apolipoprotein_a': 'Apolipoprotein A',
                                               'cystatin_c': 'Cystatin C', 'erythrocyte_count': 'Erythrocyte count',
                                               'hdl_cholesterol': 'HDL Cholesterol', 'platelet_volume': 'Platelet volume',
                                               'shbg': 'SHBG', 'standing_height': 'Standing height',
                                               'triglycerides': 'Triglycerides'})
    df['type'] = df['type'].replace({'centralized': 'Centralized + centralized GWAS', 'covariates_only': 'Covariates only',
                                     'federated': 'Federated + meta-GWAS', 'local-globaltestset': 'Local + local GWAS',
                                    'local-globaltestset-meta': 'Local + meta-GWAS'})
    fig = px.scatter(df, x='dataset', facet_col='phenotype', y='median', facet_col_wrap=5,
                     error_y='error_up', error_y_minus='error_down',
                     size='sample_count', color='type', size_max=16,
                     facet_col_spacing=0,
                     facet_row_spacing=0,
                     opacity=0.3,
                     width=1200, height=1000)

    fig.update_xaxes(categoryorder='array',
                     categoryarray=df_local_globaltestset[['tags.dataset', 'tags.sample_count']].drop_duplicates()
                     .sort_values('tags.sample_count')['tags.dataset'].tolist() + ['federated'],
                     tick0=0.0, dtick=1.0,
                     showticklabels=False, title=None)
    fig.update_yaxes(tick0=0.0, dtick=0.05,
                     showticklabels=True,
                     matches=None,
                     title='test set R2: observed vs predicted'
                     )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
    fig.write_html(out_fn)


def continuous():
    # reading local-local: local model tested on local (native) test set
    df_local_localtestset = read_local_local(exp_name=EXPERIMENT_NAME_DICT['continuous']['local']['local_local'],
                                             exp_type='local-localtestset',
                                             selected_metrics=SELECTED_METRICS)
    df_meta_localtestset = read_local_local(exp_name=EXPERIMENT_NAME_DICT['continuous']['meta']['local_local'],
                                            exp_type='local-localtestset-meta',
                                            selected_metrics=SELECTED_METRICS)

    # reading local-global: local model tested on global test set
    df_local_globaltestset = read_local_global(exp_name=EXPERIMENT_NAME_DICT['continuous']['local']['local_global'],
                                               exp_type='local-globaltestset',
                                               selected_metrics=SELECTED_METRICS)

    df_meta_globaltestset = read_local_global(exp_name=EXPERIMENT_NAME_DICT['continuous']['meta']['local_global'],
                                               exp_type='local-globaltestset-meta',
                                               selected_metrics=SELECTED_METRICS)

    # reading centralized model tested on global test set
    df_centr = read_centr(exp_name=EXPERIMENT_NAME_DICT['continuous']['local']['centr'], 
                          exp_type='centralized',
                          selected_metrics=SELECTED_METRICS)

    df_centr_meta = read_centr(exp_name=EXPERIMENT_NAME_DICT['continuous']['meta']['centr'], 
                          exp_type='centralized-meta',
                          selected_metrics=SELECTED_METRICS)

    # reading federated model tested on global test set
    df_feder = read_feder(exp_name=EXPERIMENT_NAME_DICT['continuous']['federated'], selected_metrics=SELECTED_METRICS)

    df_cov = read_covariates_only(exp_name=EXPERIMENT_NAME_DICT['continuous']['cov_only'], selected_metrics=SELECTED_METRICS)

    # imputing missing tags
    df_local_globaltestset, df_meta_globaltestset, df_feder, df_cov = impute_missing_columns(df_local_globaltestset=df_local_globaltestset,
                                                              df_meta_globaltestset=df_meta_globaltestset,
                                                              df_feder=df_feder,
                                                              df_local_localtestset=df_local_localtestset,
                                                              df_centr=df_centr,
                                                              df_cov=df_cov)

    df = prepare_data_to_plot(df_local_localtestset=df_local_localtestset,
                              df_meta_localtestset=df_meta_localtestset,
                              df_local_globaltestset=df_local_globaltestset,
                              df_meta_globaltestset=df_meta_globaltestset,
                              df_centr=df_centr,
                              df_centr_meta=df_centr_meta,
                              df_feder=df_feder,
                              df_cov=df_cov,
                              metric_name='metrics.test_r2')

    return df, df_local_globaltestset


# Todo: adapt for plotting everything together (copy over changes from continuous())
def binary():
    # reading local-local: local model tested on local (native) test set
    df_local_localtestset = read_local_local(exp_name=EXPERIMENT_NAME_DICT['binary'][snps]['local_local'],
                                             selected_metrics=['metrics.train_roc_auc', 'metrics.test_roc_auc'])

    # reading local-global: local model tested on global test set
    df_local_globaltestset = read_local_global(exp_name=EXPERIMENT_NAME_DICT['binary'][snps]['local_global'],
                                             selected_metrics=['metrics.train_roc_auc', 'metrics.test_roc_auc'])

    # reading centralized model tested on global test set
    df_centr = read_centr(exp_name=EXPERIMENT_NAME_DICT['binary'][snps]['centr'],
                                             selected_metrics=['metrics.train_roc_auc', 'metrics.test_roc_auc'])

    # reading federated model tested on global test set
    df_feder = read_feder(exp_name=EXPERIMENT_NAME_DICT['binary']['federated'],
                                             selected_metrics=['metrics.train_auc', 'metrics.test_auc'])
    df_feder['metric_name'] = df_feder['metric_name'].replace({'metrics.train_auc': 'metrics.train_roc_auc',
                                                               'metrics.test_auc': 'metrics.test_roc_auc'})


    df_cov = read_covariates_only(exp_name=EXPERIMENT_NAME_DICT['binary']['cov_only'],
                                  selected_metrics=['metrics.train_roc_auc', 'metrics.test_roc_auc'])

    # imputing missing tags
    df_local_globaltestset, df_feder, df_cov = impute_missing_columns(df_local_globaltestset=df_local_globaltestset,
                                                              df_feder=df_feder,
                                                              df_local_localtestset=df_local_localtestset,
                                                              df_centr=df_centr,
                                                              df_cov=df_cov)

    df = prepare_data_to_plot(df_local_localtestset=df_local_localtestset,
                              df_local_globaltestset=df_local_globaltestset,
                              df_centr=df_centr,
                              df_feder=df_feder,
                              df_cov=df_cov,
                              metric_name='metrics.test_roc_auc')

    return df, df_local_localtestset


if __name__ == '__main__':
    df, df_local_globaltestset = continuous()

    plot(df, df_local_globaltestset, out_fn='/home/genxadmin/tmp_continuous.html')

    # to plot binary pheotypes:    
    # df2, _ = binary()
    # plot(pd.concat([df, df_2]), b, out_fn='ayy.html')

