import csv

import numpy as np


class PCAExperimentsData:
    """
    Structure of the PCA experiments log file.
    """
    FIELDNAMES = ['pruning_parameter', 'SNPs', 'train_accuracy', 'val_accuracy', 'test_accuracy']

    def __init__(self, metrics_filename, metrics):
        self.metrics_filename = metrics_filename
        self.load(metrics)

    def load(self, metrics):
        """
        Loads PCA metrics from the the log files of the following structure:

        # Header
        pruning_parameter SNPs train_accuracy val_accuracy test_accuracy
        """
        self.data = {}

        with open(self.metrics_filename, 'r') as metrics_file:
            metrics_file.readline() # skip header
            reader = csv.DictReader(
                metrics_file,
                delimiter='\t',
                fieldnames=PCAExperimentsData.FIELDNAMES
            )

            for row in reader:
                pruning_parameter = row['pruning_parameter']
                if pruning_parameter not in self.data:
                    self.data[pruning_parameter] = {
                        'SNPs': int(row['SNPs']),
                        'train_accuracy': [],
                        'val_accuracy': [],
                        'test_accuracy': []
                    }

                for metric in metrics:
                    self.data[pruning_parameter][metric].append(float(row[metric]))

    def extend_with_median(self, metrics):
        for pruning_parameter in self.data:
            for metric in metrics:
                values = self.data[pruning_parameter][metric]
                self.data[pruning_parameter][f'{metric}_median'] = np.median(values)

    def extend_with_bands(self, metrics, quantile=80):
        for pruning_parameter in self.data:
            for metric in metrics:
                values = self.data[pruning_parameter][metric]
                self.data[pruning_parameter][f'{metric}_lower'] = np.percentile(values, (100 - quantile) / 2)
                self.data[pruning_parameter][f'{metric}_upper'] = np.percentile(values, (100 + quantile) / 2)

    def get(self, metric):
        pruning_parameters = sorted(list(self.data.keys()))

        snps, values = [], []
        for pruning_parameter in pruning_parameters:
            snps.append(self.data[pruning_parameter]['SNPs'])
            values.append(self.data[pruning_parameter][metric])

        return snps, values

    def __str__(self):
        return self.data.__str__()
