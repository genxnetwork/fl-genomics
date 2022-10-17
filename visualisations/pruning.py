# TODO: selected pruing parameter: 70k SNPs

import csv
import statistics

import numpy as np
import plotly.graph_objs as go

from collections import defaultdict


CENTRALIZED_PCA_METRICS = '/Users/misha/dev/FL/Pruning/centralized-pca-pruning-accuracy.log'
FEDERATED_PCA_FILENAME = '/Users/misha/dev/FL/Pruning/federated-pca-pruning-accuracy.log'


def plot_main_line(x, y, legendrank, color, name, dash, use_spline=False):
    line = dict(color=color, width=3, dash=dash)
    if use_spline:
        line['shape'] = 'spline'
    # marker=dict(size=8, line=dict(width=1, color='black'))
    return go.Scatter(
        x=x, y=y, line=line,
        # mode='lines',
        mode='markers+lines',
        name=name, legendrank=legendrank,
        # marker=marker
    )


def plot_error_band(x, y_lower, y_upper, legendrank, fillcolor, name, use_pattern=False, use_spline=False):
    x = x + x[::-1]
    y = y_upper + y_lower[::-1]
    line = dict(color='rgba(0, 0, 0, 0)')
    if use_spline:
        line['shape'] = 'spline'
    # fillpattern = dict(shape='x', size=4, solidity=0.2, fgcolor='rgb(94, 98, 118)') if use_pattern else dict()
    fillpattern = dict(shape='x', size=4, solidity=0.4, fgcolor='white') if use_pattern else dict()
    return go.Scatter(
        x=x, y=y, line=line, fill='toself', fillcolor=fillcolor,
        hoverinfo='skip', name=name, legendrank=legendrank, fillpattern=fillpattern
    )


def median_min_max(x):
    return statistics.median(x), min(x), max(x)


def median_percentile(x, quantile=80):
    return statistics.median(x), np.percentile(x, (100 - quantile) / 2), np.percentile(x, (100 + quantile) / 2)


def mean_percentile(x, quantile=90):
    return np.mean(x), np.percentile(x, (100 - quantile) / 2), np.percentile(x, (100 + quantile) / 2)


def load_metics_file(filename):
    metrics = {
        'snps': {},
        'train': defaultdict(list),
        'validation': defaultdict(list),
        'test': defaultdict(list)
    }

    with open(filename, 'r') as metrics_file:
        reader = csv.reader(metrics_file, delimiter=' ')
        next(reader)
        for row in reader:
            pruning_parameter   = float(row[0])
            snps_number         = float(row[1])
            train_accuracy      = float(row[2])
            validation_accuracy = float(row[3])
            test_accuracy       = float(row[4])

            metrics['snps'][pruning_parameter] = snps_number
            metrics['train'][pruning_parameter].append(train_accuracy)
            metrics['validation'][pruning_parameter].append(validation_accuracy)
            metrics['test'][pruning_parameter].append(test_accuracy)

    metrics['pruning_parameter'] = list(metrics['train'].keys())
    metrics['pruning_parameter'].sort()

    for part in ['train', 'validation', 'test']:
        metrics[f'{part}_middle'] = []
        metrics[f'{part}_lower'] = []
        metrics[f'{part}_upper'] = []

        for pruning_parameter in metrics['pruning_parameter']:
            # middle, lower, upper = mean_percentile(metrics[part][pruning_parameter], quantile=80)
            # middle, lower, upper = median_min_max(metrics[part][pruning_parameter])
            middle, lower, upper = median_percentile(metrics[part][pruning_parameter], quantile=80)
            metrics[f'{part}_middle'].append(middle)
            metrics[f'{part}_lower'].append(lower)
            metrics[f'{part}_upper'].append(upper)

    return metrics


if __name__ == '__main__':
    plots = []

    metrics_cnt = load_metics_file(CENTRALIZED_PCA_METRICS)
    metrics_fed = load_metics_file(FEDERATED_PCA_FILENAME)
    snps_number_cnt = [metrics_cnt['snps'][parameter] for parameter in metrics_cnt['pruning_parameter']]
    snps_number_fed = [metrics_fed['snps'][parameter] for parameter in metrics_fed['pruning_parameter']]

    # Centralized, bands
    for part, legendrank, fillcolor in zip(
        ['train', 'validation', 'test'],
        [4, 5, 6],
        ['rgba(76, 156, 192, 0.3)', 'rgba(237, 149, 74, 0.3)', 'rgba(122, 183, 75, 0.3)']
    ):
        name = f'Centralized PCA, {part} (band)'
        plots.append(
            plot_error_band(
                snps_number_cnt, metrics_cnt[f'{part}_lower'], metrics_cnt[f'{part}_upper'],
                legendrank=legendrank, fillcolor=fillcolor, name=name, use_pattern=True,
                use_spline=False
            )
        )

    # Federated, bands
    for part, legendrank, fillcolor in zip(
        ['train', 'validation', 'test'],
        [10, 11, 12],
        ['rgba(76, 156, 192, 0.3)', 'rgba(237, 149, 74, 0.3)', 'rgba(122, 183, 75, 0.3)'],
    ):
        name = f'Federated PCA, {part} (band)'
        plots.append(
            plot_error_band(
                snps_number_fed, metrics_fed[f'{part}_lower'], metrics_fed[f'{part}_upper'],
                legendrank=legendrank, fillcolor=fillcolor, name=name, use_pattern=False,
                use_spline=False
            )
        )

    # Centralized, lines
    for part, legendrank, color in zip(
        ['train', 'validation', 'test'],
        [1, 2, 3],
        ['rgb(46, 126, 162)', 'rgb(227, 129, 44)', 'rgb(92, 153, 45)'],
    ):
        name = f'Centralized PCA, {part} (median)'
        plots.append(
            plot_main_line(
                snps_number_cnt, metrics_cnt[f'{part}_middle'],
                legendrank=legendrank, color=color, name=name, dash='dot',
                use_spline=False
            )
        )

    # Federated, lines
    for part, legendrank, color in zip(
        ['train', 'validation', 'test'],
        [7, 8, 9],
        ['rgb(46, 126, 162)', 'rgb(227, 129, 44)', 'rgb(92, 153, 45)'],
    ):
        name = f'Federated PCA, {part} (median)'
        plots.append(
            plot_main_line(
                snps_number_fed, metrics_fed[f'{part}_middle'],
                legendrank=legendrank, color=color, name=name, dash='solid',
                use_spline=False
            )
        )

    xrange = [
        min(
            min(snps_number_cnt),
            min(snps_number_fed)
        ),
        100000
    ]

    fig = go.Figure(plots)
    fig.update_xaxes(title_text='Number of SNPs', range=xrange)
    # fig.update_xaxes(title_text='Number of SNPs (log)', type='log')
    fig.update_yaxes(title_text='Accuracy')
    fig.update_layout(title_text='Centralized MLP Accuracy for Centralized and Federated PCA')
    fig.show()
