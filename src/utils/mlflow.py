import logging

import mlflow
import numpy as np
import pandas as pd


def folds_to_quantiles(dfc, fold_col='tags.fold_index', val_col='metric_value', q=dict(lower=0.1, median=0.5, upper=0.9)):
    print(2+2)
    df = dfc.groupby([x for x in dfc.columns if (x != fold_col) & (x != val_col)])[val_col].quantile(list(q.values())).unstack()
    return df


def mlflow_get_results_table(client, experiment_name, selected_tags, selected_metric,
                             rename_cols={},
                             cols_to_int=[],
                             custom_transform=None):
    """ Retrieve local models from an mlflow client """
    exp_id = client.get_experiment_by_name(experiment_name).experiment_id
    df = mlflow.search_runs(experiment_ids=[exp_id]).rename(columns=rename_cols)
    df[cols_to_int] = df[cols_to_int].astype(int)
    if custom_transform is not None:
        df = custom_transform(df)

    # melting dataframe to move from (metric1, metric2, ...) columns to (metric_name, metric_value) columns
    # for easier visualisation
    df = df.set_index(selected_tags)[selected_metric]
    if not df.index.is_unique:
        print('Warning! Runs with duplicated tags are found!')
    ser = df.stack()
    ser.index.names = ser.index.names[:-1] + ['metric_name']
    ser.name = 'metric_value'
    return ser.reset_index()

def assemble_metric_history(client, experiment_name, selected_tags, per_epoch_metrics, per_round_metrics,
                            rename_cols={}):
    all_metrics = per_epoch_metrics + per_round_metrics
    exp_id = client.get_experiment_by_name(experiment_name).experiment_id
    runs_df = mlflow.search_runs(experiment_ids=[exp_id]).rename(columns=rename_cols)
    runs_df['id'] = runs_df['tags.mlflow.parentRunId'].fillna(runs_df['run_id'])
    runs_df['params.name'] = runs_df['params.name'].fillna('server')
    runs_df['total_rounds'] = runs_df['params.server'].fillna(method='backfill')
    runs_df['params.scheduler'] = runs_df['params.scheduler'].fillna(method='backfill')
    runs_df['total_rounds'] = [eval(i).get('rounds') for i in runs_df['total_rounds']]
    runs_df['epochs_in_round'] = [eval(i).get('epochs_in_round') for i in runs_df['params.scheduler']]
    runs_df['run_name'] = runs_df['total_rounds'].astype(str) + ', ' + runs_df['params.name']
    logging.info('extracting metrics')
    metrics_data = [[{i.step: i.value for i in client.get_metric_history(run_id=run_id, key=key)} for key in all_metrics] for run_id in runs_df['run_id'].to_list()]
    logging.info('done!')
    df = pd.DataFrame(metrics_data, columns=pd.Index(all_metrics, name='metric_name'),
                      index=runs_df.set_index(list(set(['run_id', 'id', 'run_name', 'total_rounds', 'epochs_in_round',
                                                        'params.name'] + selected_tags))).index)
    ser = df.stack()
    total_epochs = ser.apply(len).groupby('id').max()
    total_epochs.name = 'total_epochs'
    df = ser.to_frame().join(total_epochs).set_index('total_epochs', append=True)
    df = pd.DataFrame([pd.Series(d) for d in df[0].values], index = df.index).sort_index(axis=1)
    df.columns.name = 'epoch'
    df = df.stack().reset_index().rename(columns={0: 'metric_value'})
    return df[['params.name', 'total_rounds', 'epoch', 'metric_name', 'metric_value']]
