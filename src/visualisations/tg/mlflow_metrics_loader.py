import os
import json

import pandas as pd


class MLflowMetricsLoader:
    def __init__(self, directory):
        """Load metrics from a bunch of runs corresponding to a single fold."""
        self.directory = directory
        self.parent_run, self.child_runs = self.get_run_ids()
        self.strategy, self.epochs_in_round = self.get_strategy()
        self.trace = self.get_trace()

    def get_run_ids(self):
        runs = next(os.walk(self.directory))[1]
        parent_run, child_runs = None, []

        for run in runs:
            mlflow_parent_run_id = os.path.join(self.directory, run, 'tags', 'mlflow.parentRunId')
            if os.path.exists(mlflow_parent_run_id):
                child_runs.append(run)
            else:
                parent_run = run

        return parent_run, child_runs

    def get_node(self, run):
        if run == self.parent_run:
            return 'ALL'
        else:
            node_filename = os.path.join(self.directory, run, 'params', 'index')
            with open(node_filename) as node_file:
                return node_file.read().strip()

    def get_trace(self):
        fold_filename = os.path.join(self.directory, self.parent_run, 'tags', 'fold')
        if not os.path.isfile(fold_filename):
            # Centralized runs have another tag for the fold index =(
            fold_filename = os.path.join(self.directory, self.parent_run, 'tags', 'fold_index')

        with open(fold_filename) as fold_file:
            fold_index = fold_file.readline().strip()
            return f'fold-{fold_index}'

    @staticmethod
    def jsonify(text):
        text = text.replace('\'', '"')
        text = text.replace('True', 'true')
        text = text.replace('False', 'false')
        return text

    @staticmethod
    def is_scalar(metric):
        return 'test' in metric

    def get_strategy(self):
        strategy = os.path.join(self.directory, self.parent_run, 'params', 'strategy')

        if not os.path.isfile(strategy):
            return 'centralized', 1

        with open(strategy) as strategy_file:
            strategy_json = self.jsonify(strategy_file.read())
            strategy_name = json.loads(strategy_json)['name']

        scheduler = os.path.join(self.directory, self.parent_run, 'params', 'scheduler')
        with open(scheduler) as scheduler_file:
            scheduler_json = self.jsonify(scheduler_file.read())
            scheduler_dict = json.loads(scheduler_json)
            epochs_in_round = scheduler_dict['epochs_in_round']

        return f'{strategy_name}-{epochs_in_round}', epochs_in_round

    def load_metrics_array(self, name, run):
        filename = os.path.join(self.directory, run, 'metrics', name)

        values = pd.read_csv(filename, sep=' ', header=None)
        values.columns = ['timestamp', name, 'epoch']
        values = values.loc[~values[['timestamp', 'epoch']].duplicated(), :].sort_values('timestamp')
        values = values[:-1]

        return values['epoch'].to_numpy(), values[name].to_numpy()

    def load_metric_scalar(self, name, run):
        filename = os.path.join(self.directory, run, 'metrics', name)
        with open(filename) as metric_file:
            return float(metric_file.read().strip().split(' ')[1])

    def load_metrics(self, metrics, run):
        node = self.get_node(run)
        data = {'strategy': self.strategy, 'trace': self.trace, 'node': node}

        for metric in metrics:
            if self.is_scalar(metric):
                value = self.load_metric_scalar(metric, run)
                data[metric] = value
            else:
                epoch, values = self.load_metrics_array(metric, run)
                data[metric] = values
                if 'epoch' in data:
                    continue

                if node == 'ALL':
                    data['round'] = epoch
                    data['epoch'] = epoch * self.epochs_in_round
                else:
                    data['epoch'] = epoch

        return data

    def load(self, metrics, parent_only=True):
        data = []
        data.append(self.load_metrics(metrics, self.parent_run))

        if parent_only:
            return data

        for run in self.child_runs:
            data.append(self.load_metrics(metrics, run))

        return data
