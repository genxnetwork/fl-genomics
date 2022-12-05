import os
import math

import pandas as pd
import plotly.express.colors as px_colors

from argparse import ArgumentParser
from experiments_data import ExperimentsData
from pca_experiments_data import PCAExperimentsData
from loss_plot import LossPlot
from single_node_plot import SingleNodePlot
from accuracy_plot import AccuracyPlot
from accuracy_on_pruning_plot import AccuracyOnPruningPlot


GENXT_COLORS = [
    'rgb(46, 126, 162)',
    'rgb(227, 129, 44)',
    'rgb(92, 153, 45)'
]


GENXT_COLORS_DIMMED = [
    'rgba(76, 156, 192, 0.3)',
    'rgba(237, 149, 74, 0.3)',
    'rgba(122, 183, 75, 0.3)'
]


PLOTLY_COLORS_DIMMED = [
    'rgba(99, 110, 250, 0.25)',
    'rgba(239, 85, 59, 0.25)',
    'rgba(0, 204, 150, 0.25)'
]


"""
Communication cost value is proportional to the number of rounds, i.e.
communication_cost = `k` * `rounds`. Coefficient `k` equals to 2 * `n_node` (=5) * `mode_size`.
We suppose that model_size does not change between rounds. It may be found in two different ways.

[1] Use pickle library. For that, a model is exported using the following piece of code:
```
import pickle

experiment.load_data()
experiment.create_model()
    with open('model.pickle', 'wb') as model_file:
    pickle.dump(experiment.model, model_file)
```
It gives model_size = 722KB.

[2] Multiply number of model parameters to sizeof(float32) = 32 bit.
`model_size` = (20 * 800 + 800 * 200 + 200 * 26) * 32 (bit) = 708KB.

Both [1] and [2] give the estimation `k` ~ 7MB.
"""
COMMUNICATION_COST_PER_ROUND = 7 # MB


"""
Communication costs per variant for the plink + P-STACK federated PCA algorithm.
"""
COMMUNICATION_COST_PER_SNP = 0.039 # MB


class Command:
    VIEW_EXPERIMENTS_DATA = 'view-experiments-data'
    COMPUTE_MEDIAN_TEST_ACCURACY = 'compute-median-test-accuracy'
    PLOT_TRAIN_LOSS = 'plot-train-loss'
    PLOT_VALIDATION_LOSS = 'plot-validation-loss'
    PLOT_SINGLE_NODE_TRAIN_LOSS = 'plot-single-node-train-loss'
    PLOT_VALIDATION_ACCURACY_WITH_COST = 'plot-validation-accuracy-with-cost'
    PLOT_TRAIN_ACCURACY_WITH_COST = 'plot-train-accuracy-with-cost'
    PLOT_FEDERATED_PCA_ACCURACY = 'plot-federated-pca-accuracy'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--directory-pca', dest='directory_pca') # for federated PCA only
    parser.add_argument('--directory-16k-epochs', dest='directory_16k_epochs')
    parser.add_argument('--directory-4k-rounds', dest='directory_4k_rounds')
    parser.add_argument('--add-trace', dest='add_trace')
    parser.add_argument('--export', dest='export_filename')
    args = parser.parse_args()

    def get_single_directory():
        directory = args.directory_16k_epochs if args.directory_16k_epochs is not None \
            else args.directory_4k_rounds

        assert directory is not None
        return directory

    if args.command == Command.VIEW_EXPERIMENTS_DATA:
        experiments_data = ExperimentsData(
            get_single_directory(),
            metrics=['val_accuracy', 'val_loss', 'test_accuracy'],
            parent_only=False
        )

        print(experiments_data)

    elif args.command in Command.COMPUTE_MEDIAN_TEST_ACCURACY:
        experiments_data = ExperimentsData(
            get_single_directory(),
            metrics=['test_accuracy'],
            parent_only=True
        )
        experiments_data.extend_with_median('test_accuracy')

        out = []
        for strategy in experiments_data.get_strategies():
            value = experiments_data.get(strategy, 'ALL', 'median', 'test_accuracy')
            result = '%.3f' % round(value, 3)
            out.append(f'{strategy}: {result}')

        print('\n'.join(out))

    elif args.command in [Command.PLOT_VALIDATION_LOSS, Command.PLOT_TRAIN_LOSS]:
        if args.command == Command.PLOT_VALIDATION_LOSS:
            metric, yaxis = 'val_loss', 'Validation loss'
        else:
            metric, yaxis = 'train_loss', 'Train loss'

        experiments_data = ExperimentsData(get_single_directory(), metrics=[metric], parent_only=True)
        experiments_data.extend_with_median(metric)

        plot = LossPlot(yaxis=yaxis, colors=px_colors.qualitative.Plotly)
        plot.add_centralized_model_trace(
            epoch=experiments_data.get('centralized', 'ALL', 'median', 'epoch'),
            loss=experiments_data.get('centralized', 'ALL', 'median', metric),
        )

        strategies = experiments_data.get_strategies()
        strategies.remove('centralized')
        for strategy in strategies:
            plot.add_federated_model_trace(
                epoch=experiments_data.get(strategy, 'ALL', 'median', 'epoch'),
                loss=experiments_data.get(strategy, 'ALL', 'median', metric),
                name=strategy.capitalize()
            )
        if args.add_trace is not None:
            trace_df = pd.read_csv(args.add_trace, header=None, sep=' ')
            trace_df = trace_df[trace_df[2] > 0]
            plot.add_model_trace(epoch=trace_df[2], loss=trace_df[1], name='custom', line={'color': 'black', 'width': 3})
            # tracedf2 = pd.read_csv('/home/genxadmin/mlflow_homo/multirun/2022-12-01/19-00-50/0/mlruns/1/003e6078f72d43d8930c290a1756028b/metrics/val_loss', header=None, sep=' ')
            # tracedf2 = tracedf2[tracedf2[2] > 0].sort_values(2)
            # plot.add_model_trace(epoch=tracedf2[2], loss=tracedf2[1], name='scaffold_homo', line={'color': 'brown', 'width': 3})
        if args.export_filename is not None:
            plot.export(args.export_filename)
        else:
            plot.show()

    elif args.command == Command.PLOT_SINGLE_NODE_TRAIN_LOSS:
        experiments_data = ExperimentsData(get_single_directory(), metrics=['train_loss'], parent_only=False)

        yaxis = 'Train loss'
        color = px_colors.qualitative.Plotly[6]

        plot = SingleNodePlot(yaxis='Train loss', title='Learning curve, node=AMR, fold=5', color=color)
        plot.add_node_trace(
            epoch=experiments_data.get('fedavg-32', 'AMR', 'fold-5', 'epoch'),
            loss=experiments_data.get('fedavg-32', 'AMR', 'fold-5', 'train_loss')
        )

        plot.set_right_yrange(0.025, 0.09)
        plot.set_right_xrange(14990, 15160)
        plot.set_right_ticks(tickvals=[15008, 15040, 15072, 15104, 15136])

        if args.export_filename is not None:
            plot.export(args.export_filename)
        else:
            plot.show()

    elif args.command in [Command.PLOT_TRAIN_ACCURACY_WITH_COST, Command.PLOT_VALIDATION_ACCURACY_WITH_COST]:
        # Here both of the direcories are required
        assert args.directory_16k_epochs is not None
        assert args.directory_4k_rounds is not None

        if args.command == Command.PLOT_VALIDATION_ACCURACY_WITH_COST:
            metric, yaxis = 'val_accuracy', 'Validation accuracy'
        else:
            metric, yaxis = 'train_accuracy', 'Train accuracy'

        experiments_data_16k_epochs = ExperimentsData(
            args.directory_16k_epochs,
            metrics=[metric],
            parent_only=True
        )

        experiments_data_4k_rounds = ExperimentsData(
            args.directory_4k_rounds,
            metrics=[metric],
            parent_only=True
        )

        experiments_data_16k_epochs.extend_with_median(metric)
        experiments_data_4k_rounds.extend_with_median(metric)

        plot = AccuracyPlot(
            yaxis=yaxis,
            colors=px_colors.qualitative.Plotly,
            cost_per_round_mb=COMMUNICATION_COST_PER_ROUND
        )

        for strategy in ['fedavg-1', 'fedavg-2', 'fedavg-4']:
            plot.add_federated_model_trace(
                epoch=experiments_data_16k_epochs.get(strategy, 'ALL', 'median', 'epoch'),
                round=experiments_data_16k_epochs.get(strategy, 'ALL', 'median', 'round'),
                accuracy=experiments_data_16k_epochs.get(strategy, 'ALL', 'median', metric),
                name=strategy
            )

        for strategy in ['fedavg-8', 'fedavg-16', 'fedavg-32']:
            plot.add_federated_model_trace(
                epoch=experiments_data_4k_rounds.get(strategy, 'ALL', 'median', 'epoch'),
                round=experiments_data_4k_rounds.get(strategy, 'ALL', 'median', 'round'),
                accuracy=experiments_data_4k_rounds.get(strategy, 'ALL', 'median', metric),
                name=strategy
            )

        cost_ticks = [
            (    0,    '0 GB'),
            ( 3500,  '3.5 GB'),
            ( 7000,    '7 GB'),
            (10500, '10.5 GB'),
            (14000,   '14 GB'),
            (17500, '17.5 GB'),
            (21000,   '21 GB'),
            (24500, '24.5 GB'),
            (28000,   '28 GB')
        ]

        plot.set_pca_costs(2524) # MB
        plot.set_yrange(0.15, 0.9)
        plot.set_left_xrange(0, 16384)
        plot.set_right_xrange(0, 4096)
        plot.set_right_ticks(
            round_tickvals=[0, 1000, 2000, 3000, 4000],
            cost_tickvals=[val for val, _ in cost_ticks],
            cost_ticktext=[text for _, text in cost_ticks]
        )

        if args.export_filename is not None:
            plot.export(args.export_filename)
        else:
            plot.show()

    elif args.command == Command.PLOT_FEDERATED_PCA_ACCURACY:
        # Possible metrics: ['train_accuracy', 'val_accuracy', 'test_accuracy']
        metrics = ['train_accuracy', 'val_accuracy']

        # These names are used for the plot legend
        names = {
            'train_accuracy': 'Training accuracy',
            'val_accuracy': 'Validation accuracy',
            'test_accuracy': 'Test accuracy'
        }

        centralied_pca_metrics_filename = os.path.join(args.directory_pca, 'centralized-pca-metrics.tsv')
        centralized_experiments_data = PCAExperimentsData(centralied_pca_metrics_filename, metrics)
        centralized_experiments_data.extend_with_median(metrics=metrics)
        centralized_experiments_data.extend_with_bands(metrics=metrics)

        federated_pca_metrics_filename = os.path.join(args.directory_pca, 'federated-pca-metrics.tsv')
        federated_experiments_data = PCAExperimentsData(federated_pca_metrics_filename, metrics)
        federated_experiments_data.extend_with_median(metrics=metrics)
        federated_experiments_data.extend_with_bands(metrics=metrics)

        plot = AccuracyOnPruningPlot(
            trace_colors=px_colors.qualitative.Plotly,
            band_colors=PLOTLY_COLORS_DIMMED,
            cost_per_snp_mb=COMMUNICATION_COST_PER_SNP
        )

        min_snps = math.inf
        max_snps = 0
        for metric in metrics:
            snps, median = centralized_experiments_data.get(f'{metric}_median')
            _, lower = centralized_experiments_data.get(f'{metric}_lower')
            _, upper = centralized_experiments_data.get(f'{metric}_upper')

            min_snps = min(min_snps, min(snps))
            max_snps = max(max_snps, max(snps))
            plot.add_centralized_model_trace(snps, median, name=f'{names[metric]}, centralized PCA')
            plot.add_centralized_model_band(snps, lower, upper, name=f'{names[metric]}, centralized PCA')

            snps, median = federated_experiments_data.get(f'{metric}_median')
            _, lower = federated_experiments_data.get(f'{metric}_lower')
            _, upper = federated_experiments_data.get(f'{metric}_upper')

            min_snps = min(min_snps, min(snps))
            max_snps = max(max_snps, max(snps))
            plot.add_federated_model_trace(snps, median, name=f'{names[metric]}, federated PCA')
            plot.add_federated_model_band(snps, lower, upper, name=f'{names[metric]}, federated PCA')

        snps_tickvals = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
        cost_ticktext = ['%.2f GB' % round(val * COMMUNICATION_COST_PER_SNP / 1024, 2) for val in snps_tickvals]

        plot.set_selected_variants_number(64636)
        plot.set_yrange(0.74, 0.94)
        plot.set_xrange(min_snps, max_snps)
        plot.set_xticks(
            snps_tickvals=snps_tickvals,
            cost_tickvals=snps_tickvals,
            cost_ticktext=cost_ticktext
        )

        if args.export_filename is not None:
            plot.export(args.export_filename)
        else:
            plot.show()
