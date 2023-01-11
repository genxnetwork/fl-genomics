import os

import numpy as np

from mlflow_metrics_loader import MLflowMetricsLoader


class ExperimentsData:
    """
    Load multiple experiments MLflow data into a single ditionary representation.
    """

    def __init__(self, directory, metrics, parent_only=True):
        self.directory = directory
        self.load(metrics, parent_only=parent_only)

    def load(self, metrics, parent_only=True):
        """
        data[strategy][node][trace] = {
            'epoch': np.ndarray,
            'train_loss': np.ndarray,
            'train_accuracy': np.ndarray
            'test_accuracy': float
        }

        strategy = 'fedavg-1' | 'fedavg-2' | ... | 'centralized'
        trace    = 'fold-0' | 'fold-1' | ... | 'median' | 'lower' | 'upper'
        node     = 'ALL' | 'AFR' | 'AMR' | 'EUR' | 'EAS' | 'SAS'
        """
        self.data = {}

        experiments = next(os.walk(self.directory))[1]
        for experiment in experiments:
            experiment_directory = os.path.join(self.directory, experiment)
            for data in MLflowMetricsLoader(experiment_directory).load(metrics, parent_only=parent_only):
                strategy, node, trace = data['strategy'], data['node'], data['trace']

                if strategy not in self.data:
                    self.data[strategy] = {}
                if node not in self.data[strategy]:
                    self.data[strategy][node] = {}

                self.data[strategy][node][trace] = {name: data[name] for name in metrics}
                if 'epoch' in data:
                    self.data[strategy][node][trace]['epoch'] = data['epoch']
                if 'round' in data:
                    self.data[strategy][node][trace]['round'] = data['round']

    def extend_with_median(self, metric_name):
        """
        Add 'median' to each node of all available strategies.
        """
        for strategy in self.data.keys():
            for node in self.data[strategy].keys():
                node_data = self.data[strategy][node]
                folds = list(filter(lambda x: x.startswith('fold-'), node_data.keys()))
                assert all([metric_name in node_data[fold] for fold in folds])

                if 'median' not in node_data:
                    node_data['median'] = {}

                metric_type = type(node_data[folds[0]][metric_name])
                if metric_type == float:
                    values = [node_data[fold][metric_name] for fold in folds]
                    node_data['median'][metric_name] = np.median(values)
                elif metric_type == np.ndarray:
                    values = np.vstack([node_data[fold][metric_name] for fold in folds])
                    node_data['median'][metric_name] = np.median(values, axis=0)
                    node_data['median']['epoch'] = node_data[folds[0]]['epoch']
                    if 'round' in node_data[folds[0]]:
                        node_data['median']['round'] = node_data[folds[0]]['round']

                self.data[strategy][node]


    def __str__(self):
        rows = []

        for strategy in self.data.keys():
            rows.append(strategy)
            for node in self.data[strategy].keys():
                traces = list(self.data[strategy][node].keys())
                traces.sort()
                rows.append(f'|-> {node} [{len(traces)}]: {traces}')

                metrics = self.data[strategy][node][traces[0]]
                representations = []
                for name in metrics:
                    if isinstance(metrics[name], float):
                        representations.append(name)
                    elif isinstance(metrics[name], np.ndarray):
                        representations.append(f'{name}: ({metrics[name].shape[0]})')
                representation = ', '.join(representations)
                rows.append(f'|   {representation}')

        return '\n'.join(rows)

    def get_strategies(self):
        strategies = list(self.data.keys())
        strategies = sorted(
            strategies,
            key=lambda x: 0 if '-' not in x else int(x.split('-')[1])
        )
        return strategies

    def get(self, strategy, node, trace, metric):
        return self.data[strategy][node][trace][metric]
