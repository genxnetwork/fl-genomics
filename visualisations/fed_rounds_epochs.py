from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import os
import mlflow
from mlflow.tracking import MlflowClient
import numpy
import pandas
from statsmodels.stats.weightstats import DescrStatsW
import plotly.graph_objects as go
import plotly.express as px


def plot_rounds_epochs_dependence(epochs_in_round: List[int], val_r2: List[float], test_r2: List[float], output_fn: str):
    print(val_r2)
    print(epochs_in_round)
    plt.figure(figsize=(15, 10))
    plt.plot(epochs_in_round, val_r2, marker='o', linewidth=2, markersize=8, label='Val $r^2$')
    plt.plot(epochs_in_round, test_r2, marker='o', linewidth=2, markersize=8, label='Test $r^2$')
    plt.xlabel('Epochs in FL Round', fontsize=18)
    plt.ylabel('Average $r^2$', fontsize=18)
    plt.legend(fontsize=20)
    plt.grid()
    plt.savefig(output_fn)


def weighted_ci(data: pandas.DataFrame, metric_name: str) -> Tuple[float, float]:
    weighted_stats = DescrStatsW(data[metric_name].values, weights=data['sample_count'], ddof=0)
    mean = weighted_stats.mean
    return pandas.Series([mean, mean - weighted_stats.std_mean*1.96, mean + weighted_stats.std_mean*1.96],
        index=['mean', 'ci_left', 'ci_right'])


def aggregate_run(client: MlflowClient, runs: pandas.DataFrame, metric_name: str):
    data = []
    for index, run in runs.iterrows():
        metric = client.get_metric_history(run_id=run.run_id, key=metric_name)
        run_data = numpy.array([[m.step for m in metric], [m.value for m in metric]]).T
        # print(run_data)
        max_idx = numpy.argmax(run_data[:, 1])
        # print(run_data.shape)
        run_data = run_data[[i for i in range(run_data.shape[0]) if i != max_idx]]
        run_data[:, 0] -= run_data[:, 0].min()
        # last step contains local metric of globally averaged parameters from final evaluation
        run_data = pandas.DataFrame(data=run_data[run_data[:, 0] < 62.5], columns=['step', metric_name])
        # print(run_data.shape)
        run_data.loc[:, 'run_id'] = run.run_id
        run_data.loc[:, 'sample_count'] = run['tags.sample_count']
        data.append(run_data)
    
    data = pandas.concat(data, axis='index', ignore_index=True)
    cis = data.groupby(by='step').apply(lambda x: weighted_ci(x, metric_name))
    return cis


def plot_local_run(fig: go.Figure, data: pandas.DataFrame, color: str, name: str):

    fig.add_trace(
        go.Scatter(x=data.index, y=data.ci_left,
        fill=None,
        mode='lines+markers',
        marker=dict(symbol='circle', size=5),
        line_color=color,
        line_width=2,
        name=name
    ))
    '''
    fig.add_trace(go.Scatter(
        x=data.index, y=data['mean'],
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines',
        line_color=color,
        line_width=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.ci_right,
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', 
        line_color=color,
        line_width=0.5,
    ))
    '''


def plot_global_run(fig: go.Figure, run_id: str, metric_name: str, color: str, name: str):
    
    metric = client.get_metric_history(run_id=run_id, key=metric_name)
    run_data = numpy.array([[m.step for m in metric], [m.value for m in metric]]).T
    run_data[run_data < 0] = 64
    run_data = run_data[run_data[:, 0].argsort()]
    
    fig.add_trace(
        go.Scatter(x=run_data[:, 0] - 1, y=run_data[:, 1],
        fill=None,
        mode='lines+markers',
        marker=dict(symbol='circle-x', size=8),
        line_color=color,
        line_width=2,
        name=name
    ))

def plot_all_runs(client: MlflowClient, parent_runs: pandas.DataFrame, metric_name: str, output_fn: str):
    fig = go.Figure()
    colors = px.colors.qualitative.Prism
    for i, (run_id, rounds) in enumerate(zip(parent_runs.run_id, parent_runs['params.rounds'])):
        node_runs = find_node_runs(exp_id, run_id)
        run_data = aggregate_run(client, node_runs, 'val_loss')
        color = colors[i]
        name = f'Epochs in round: {64 // rounds}'
        plot_local_run(fig, run_data, color, name)
        plot_global_run(fig, run_id, metric_name, color, 'Global: ' + name)
    
    fig.update_layout(
        title="Average Local vs Global RMSE",
        xaxis_title="Epoch",
        yaxis_title="Validation RMSE"
    )
    fig.write_html(output_fn)


def find_node_runs(exp_id: str, parent_run_id: str) -> pandas.DataFrame:
    print(f'we are searchin runs with parent {parent_run_id}')
    query = f'tags.mlflow.parentRunId = "{parent_run_id}"'
    data = mlflow.search_runs(experiment_ids=[exp_id], filter_string=query)
    return data


if __name__ == '__main__':
    CURRENT_EXPERIMENT_NAME = 'federated_lassonet_standing-height'
    client = MlflowClient()
    print(f'starting to search experiment by name: {CURRENT_EXPERIMENT_NAME}')
    exp_id = client.get_experiment_by_name(CURRENT_EXPERIMENT_NAME).experiment_id
    print(f'experiment id is {exp_id}')
    
    df = mlflow.search_runs(experiment_ids=[exp_id], filter_string='params.active_nodes = "all"')
    print(df.columns)
    print(df.head())
    df.loc[:, 'params.rounds'] = df['params.rounds'].astype(int)
    df.sort_values(by='params.rounds', ascending=True, inplace=True)
    print(df.columns)
    rounds = df['params.rounds']
    rounds = 64 // rounds
    plot_rounds_epochs_dependence(rounds.tolist(), df['metrics.val_r2'].tolist(), df['metrics.test_r2'].tolist(), 'outputs/rounds_epochs_dependence.png')
    print(f'search of runs returned {df} runs')

    plot_all_runs(client, df, 'val_loss', 'test_all_runs.html')        
    # data = parser.parse_epochs(df[0], ['val_r2'])
    # plot_rounds_epochs_dependence([0], data)