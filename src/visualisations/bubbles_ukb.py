import os
import pandas as pd

import mlflow
import plotly.express as px

from mlflow.tracking import MlflowClient

from utils.mlflow import mlflow_get_results_table, folds_to_quantiles

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

SELECTED_TAGS = ['tags.phenotype', 'tags.model', 'tags.dataset', 'tags.snp_count', 'tags.sample_count', 'tags.ethnicity',
                 'tags.fold_index']
SELECTED_METRICS = ['metrics.val_r2', 'metrics.train_r2']


def ethnicity_from_dataset(df: pd.DataFrame):
    df['tags.ethnicity'] = df['tags.dataset'].str.split('_').str[0]
    return df


if __name__ == '__main__':
    df_local = mlflow_get_results_table(client=MlflowClient(), experiment_name='local-models-local-snps',
                                  selected_tags=SELECTED_TAGS, selected_metric=SELECTED_METRICS,
                                  cols_to_int=['tags.snp_count', 'tags.sample_count'],
                                  custom_transform=ethnicity_from_dataset)

    df_centr = mlflow_get_results_table(client=MlflowClient(), experiment_name='centralized-models-centralized-snps-fixed',
                                  selected_tags=SELECTED_TAGS, selected_metric=SELECTED_METRICS,
                                  cols_to_int=['tags.snp_count', 'tags.sample_count'],
                                  custom_transform=ethnicity_from_dataset)
    df = pd.concat([df_local, df_centr])
    df = folds_to_quantiles(df.query("metric_name == 'metrics.val_r2'"), fold_col='tags.fold_index', val_col='metric_value')


    # fig = px.strip(df.query("metric_name == 'metrics.val_r2'"), x='tags.phenotype', y='metric_value')
    # fig = px.scatter(df, x='tags.dataset', facet_col='tags.phenotype', y='median')
    fig = px.bar(df, color='tags.dataset', x='tags.phenotype', y='median', barmode='group')
    fig.write_html('/home/genxadmin/tmp.html')
    pass
