import os
import pandas as pd

import mlflow
import plotly.express as px

from mlflow.tracking import MlflowClient

from utils.mlflow import mlflow_get_results_table, folds_to_quantiles

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

SELECTED_TAGS = ['tags.phenotype', 'tags.model', 'tags.dataset', 'tags.snp_count', 'tags.sample_count',
                 'tags.fold_index']
SELECTED_METRICS = ['metrics.val_r2', 'metrics.train_r2']


def ethnicity_from_dataset(df: pd.DataFrame):
    df['tags.ethnicity'] = df['tags.dataset'].str.split('_').str[0]
    return df


def normalize_by_centr_median(df: pd.DataFrame):
    df.loc[:, ['lower', 'median', 'upper']] = df.loc[:, ['lower', 'median', 'upper']] / df.loc[df['tags.dataset'].str.contains('Centralized'), 'median'].squeeze()
    return df


if __name__ == '__main__':
    df_local_localtestset = mlflow_get_results_table(client=MlflowClient(), experiment_name='local-models-local-snps',
                                                     selected_tags=SELECTED_TAGS + ['tags.node_index'], selected_metric=SELECTED_METRICS,
                                                     cols_to_int=['tags.node_index', 'tags.snp_count', 'tags.sample_count'])
    df_local_localtestset['type'] = 'local-localtestset'
    node_index_df = df_local_localtestset[['tags.node_index', 'tags.dataset']].drop_duplicates()
    samcount_index_df = df_local_localtestset[['tags.node_index', 'tags.sample_count']].drop_duplicates()

    df_local_globaltestset = mlflow_get_results_table(client=MlflowClient(), experiment_name='local-models-local-snps-centralized-metrics',
                                  selected_tags=['tags.phenotype', 'tags.fold_index', 'tags.train_node_index'], selected_metric=SELECTED_METRICS,
                                  cols_to_int=['tags.train_node_index'])
    df_local_globaltestset['type'] = 'local-globaltestset'
    df_local_globaltestset['tags.model'] = 'lassonet'
    df_local_globaltestset['tags.snp_count'] = df_local_localtestset['tags.snp_count'].iloc[0]
    df_local_globaltestset['tags.sample_count'] = df_local_globaltestset['tags.train_node_index'].replace(dict(zip(samcount_index_df['tags.node_index'], samcount_index_df['tags.sample_count'])))
    df_local_globaltestset['tags.dataset'] = df_local_globaltestset['tags.train_node_index'].replace(dict(zip(node_index_df['tags.node_index'], node_index_df['tags.dataset'])))

    df_centr = mlflow_get_results_table(client=MlflowClient(), experiment_name='centralized-models-centralized-snps-fixed',
                                  selected_tags=SELECTED_TAGS, selected_metric=SELECTED_METRICS,
                                  cols_to_int=['tags.snp_count', 'tags.sample_count'])
    df_centr['type'] = 'centralized'

    df = pd.concat([df_local_localtestset.drop(columns=['tags.node_index']),
                    df_local_globaltestset.drop(columns=['tags.train_node_index']),
                    df_centr])
    df = folds_to_quantiles(df.query("metric_name == 'metrics.val_r2'"), fold_col='tags.fold_index', val_col='metric_value')
    df = df.set_index(['tags.phenotype', 'tags.model']).groupby(['tags.phenotype', 'tags.model'])[['type', 'tags.dataset', 'tags.sample_count', 'lower', 'median',
       'upper']].apply(normalize_by_centr_median).reset_index()

    # fig = px.strip(df.query("metric_name == 'metrics.val_r2'"), x='tags.phenotype', y='metric_value')
    # fig = px.scatter(df, x='tags.dataset', facet_col='tags.phenotype', y='median')
    df['error_up'] = df['upper'] - df['median']
    df['error_down'] = df['median'] - df['lower']
    df.columns = [i.replace('tags.', '') for i in df.columns]
    # fig = px.bar(df, color='dataset', x='phenotype', y='median', barmode='group', error_y='error_up', error_y_minus='error_down')
    fig = px.scatter(df, x='dataset', facet_col='phenotype', y='median',
                     error_y='error_up', error_y_minus='error_down',
                     size='sample_count', color='type')
    fig.update_xaxes(categoryorder='array',
                     categoryarray=df[['dataset', 'sample_count']].drop_duplicates().sort_values('sample_count')['dataset'].tolist())
    fig.write_html('/home/genxadmin/tmp.html')
    pass
