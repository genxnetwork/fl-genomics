import os
import math
import mlflow
import pandas as pd
import numpy as np

from torch.nn.functional import cross_entropy
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from nn.models import MLPClassifier
from configs.phenotype_config import get_accuracy
from configs.split_config import FOLDS_NUMBER
from preprocess.pruning import PlinkPruningRunner
from tg.data_provider import DataProvider
from preprocess.federated_pca import FederatedPCASimulationRunner


LOG_FILE = '/home/genxadmin/federated-pca-metrics.tsv'
COSTS_LOG_FILE = '/home/genxadmin/federated-pca-costs.tsv'
SPLIT_DIRECTORY = '/mnt/genx-bio-share/TG/data/chip/pca_split'


def get_model(num_classes, num_features, trainer=None):
    parameters = {
        'nclass': num_classes,
        'nfeat': num_features,
        'optim_params': {
            'name': 'sgd',
            'lr': 0.1
        },
        'scheduler_params': {
            'rounds': '1',
            'epochs_in_round': 16384,
            'name': 'exponential_lr',
            'gamma': 0.9999
        },
        'loss': cross_entropy
    }

    if not trainer:
        return MLPClassifier(**parameters)
    else:
        parameters['checkpoint_path'] = trainer.checkpoint_callback.best_model_path
        return MLPClassifier.load_from_checkpoint(**parameters)


def get_trainer():
    return Trainer(
        max_epochs=16384,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=512,
                strict=False,
                verbose=False,
                mode='min'
            ),
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(monitor='val_loss', mode='min')
        ],
        precision=32,
        weights_summary='full'
    )


def write_to_logfile(pruning_threshold, variants_number, accuracy_train, accuracy_validation, accuracy_test):
    with open(LOG_FILE, 'a') as log_file:
        row = '\t'.join([
            str(pruning_threshold),
            str(variants_number),
            str(accuracy_train),
            str(accuracy_validation),
            str(accuracy_test)
        ]) + '\n'

        log_file.write(row)


def write_to_cost_logfile(variants_number, cost):
    with open(COSTS_LOG_FILE, 'a') as log_file:
        row = '\t'.join([
            str(variants_number),
            str(cost)
        ]) + '\n'

        log_file.write(row)


def get_variants_number(fold):
    filename = f'{SPLIT_DIRECTORY}/federated_pca/ALL/fold_{fold}_train.eigenvec.allele'
    allele = pd.read_csv(filename, sep='\t', header=0)
    return allele['ID'].unique().shape[0]


def compute_communication_costs(fold):
    size = 0
    for node in ['ALL', 'AFR', 'AMR', 'EAS', 'SAS', 'EUR']:
        size += os.path.getsize(f'{SPLIT_DIRECTORY}/federated_pca/{node}/fold_{fold}_train.eigenvec.allele') / (2 ** 20)

    return math.ceil(size)


def run_experiment(pruning_threshold):
    """
    Performs pruning, then runs cetralized PCA and compute model accuracy.
    """

    data_provider = DataProvider(
        f'{SPLIT_DIRECTORY}/federated_pca',
        f'{SPLIT_DIRECTORY}/only_phenotypes/ancestry',
        num_components=20,
        normalize_std=True
    )

    # Run pruning
    PlinkPruningRunner(
        source_directory=f'{SPLIT_DIRECTORY}/genotypes',
        nodes=['AFR', 'AMR', 'EAS', 'SAS', 'EUR'],
        result_filepath=f'{SPLIT_DIRECTORY}/genotypes/ALL.prune.in',
        node_filename_template='%s_filtered'
    ).run(window_size=1000, step=50, threshold=pruning_threshold)

    # Run Federated PCA
    FederatedPCASimulationRunner(
        source_folder=f'{SPLIT_DIRECTORY}/genotypes',
        result_folder=f'{SPLIT_DIRECTORY}/federated_pca',
        variant_ids_file=f'{SPLIT_DIRECTORY}/genotypes/ALL.prune.in',
        n_components=20,
        method='P-STACK',
        nodes=['AFR', 'AMR', 'EAS', 'SAS', 'EUR']
    ).run()

    for fold in range(FOLDS_NUMBER):
        # Fit model and log accuracy
        _, y_train = data_provider.load_train_data('ALL', fold)
        data_module = data_provider.create_data_module('ALL', fold)

        model = get_model(
            num_classes=len(np.unique(y_train)),
            num_features=data_provider.num_components
        )

        trainer = get_trainer()
        mlflow.set_experiment('tg')
        mlflow.start_run()
        trainer.fit(model, data_module)

        model = get_model(
            num_classes=len(np.unique(y_train)),
            num_features=data_provider.num_components,
            trainer=trainer
        )

        model.eval()
        train_loader, validation_loader, test_loader = data_module.predict_dataloader()
        y_pred, y_true = model.predict(train_loader)
        accuracy_train = get_accuracy(y_true, y_pred)

        y_pred, y_true = model.predict(validation_loader)
        accuracy_validation = get_accuracy(y_true, y_pred)

        y_pred, y_true = model.predict(test_loader)
        accuracy_test = get_accuracy(y_true, y_pred)

        variants_number = get_variants_number(fold) # should be the same for all folds
        write_to_logfile(pruning_threshold, variants_number, accuracy_train, accuracy_validation, accuracy_test)

        cost = compute_communication_costs(fold)
        write_to_cost_logfile(variants_number, cost)

        mlflow.end_run()


if __name__ == '__main__':
    with open(LOG_FILE, 'w') as log_file:
        header = '\t'.join([
            'pruning_threshold',
            'variants_number',
            'train_accuracy',
            'val_accuracy',
            'test_accuracy'
        ]) + '\n'

        log_file.write(header)

    with open(COSTS_LOG_FILE, 'w') as log_file:
        header = '\t'.join([
            'variants_number',
            'cost (MB)'
        ]) + '\n'

        log_file.write(header)

    for pruning_threshold in [
        0.001,  0.0015, 0.002,  0.0025, 0.003,  0.0035, 0.004,  0.0045, 0.005,
        0.0055, 0.006,  0.0065, 0.007,  0.0075, 0.008,  0.0085, 0.009,  0.0095,
        0.01,   0.011,  0.012,  0.013,  0.014,  0.015,  0.016,  0.017,  0.018,
        0.019,  0.02,   0.021,  0.022,  0.023,  0.024,  0.025,  0.026,  0.027,
        0.028,  0.029,  0.030
    ]:
        run_experiment(pruning_threshold)
