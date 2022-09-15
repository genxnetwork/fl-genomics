import pandas as pd
import numpy as np

from torch.nn.functional import cross_entropy
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from utils.plink import run_plink
from nn.models import MLPClassifier
from configs.phenotype_config import get_accuracy
from configs.split_config import FOLDS_NUMBER
from preprocess.pruning import PlinkPruningRunner
from tg.data_provider import DataProvider


LOG_FILE = '/home/genxadmin/centralized-pca-pruning-accuracy.log'
SPLIT_DIRECTORY = '/mnt/genx-bio-share/TG/data/chip/superpop_split'


def get_model(num_classes, num_features, trainer=None):
    parameters = {
        'nclass': num_classes,
        'nfeat': num_features,
        'optim_params': {
            'name': 'sgd',
            'lr': 0.1
        },
        'scheduler_params': {
            'rounds': '10000',
            'epochs_in_round': 1,
            'name': 'exponential_lr',
            'gamma': 0.999
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
        max_epochs=10000,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=1000,
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
        log_file.write(f'{pruning_threshold} {variants_number} {accuracy_train} {accuracy_validation} {accuracy_test}\n')


def run_experiment(pruning_threshold):
    """
    Performs pruning, then runs cetralized PCA and compute model accuracy.
    """

    data_provider = DataProvider(
        f'{SPLIT_DIRECTORY}/pca',
        f'{SPLIT_DIRECTORY}/only_phenotypes/ancestry',
        num_components=20
    )

    # Run pruning
    PlinkPruningRunner(
        source_directory=f'{SPLIT_DIRECTORY}/genotypes',
        nodes=['AFR', 'AMR', 'EAS', 'SAS', 'EUR'],
        result_filepath=f'{SPLIT_DIRECTORY}/genotypes/ALL.prune.in',
        node_filename_template='%s_filtered'
    ).run(window_size=1000, step=50, threshold=pruning_threshold)

    variants_number = len(
        pd.read_csv(f'{SPLIT_DIRECTORY}/genotypes/ALL.prune.in', sep='\t', header=None)
    )

    # Run Centralized PCA
    for fold in range(FOLDS_NUMBER):
        plink_arguments = [
            '--pfile', f'{SPLIT_DIRECTORY}/genotypes/ALL/fold_{fold}_train',
            '--extract', f'{SPLIT_DIRECTORY}/genotypes/ALL.prune.in',
            '--freq', 'counts',
            '--out',  f'{SPLIT_DIRECTORY}/pca/fold_{fold}_train_projections',
            '--pca', 'allele-wts', '20'
        ]

        run_plink(args_list=plink_arguments)

        # Project ALL only
        for part in ['train', 'val', 'test']:
            plink_arguments = [
                '--pfile', f'{SPLIT_DIRECTORY}/genotypes/ALL/fold_{fold}_{part}',
                '--extract', f'{SPLIT_DIRECTORY}/genotypes/ALL.prune.in',
                '--read-freq', f'{SPLIT_DIRECTORY}/pca/ALL/fold_{fold}_train_projections.acount',
                '--score', f'{SPLIT_DIRECTORY}/pca/ALL/fold_{fold}_train_projections.eigenvec.allele',
                    '2', '5', 'header-read', 'no-mean-imputation', 'variance-standardize',
                '--score-col-nums', '6-25',
                '--out', f'{SPLIT_DIRECTORY}/pca/ALL/fold_{fold}_{part}_projections.csv.eigenvec.sscore'
            ]

            run_plink(args_list=plink_arguments)

        # Fit model and log accuracy
        _, y_train = data_provider.load_train_data('ALL', fold)
        _, y_validation = data_provider.load_validation_data('ALL', fold)
        _, y_test = data_provider.load_test_data('ALL', fold)
        data_module = data_provider.create_data_module('ALL', fold)

        model = get_model(
            num_classes=len(np.unique(y_train)),
            num_features=data_provider.num_components
        )

        trainer = get_trainer()
        trainer.fit(model, data_module)

        model = get_model(
            num_classes=len(np.unique(y_train)),
            num_features=data_provider.num_components,
            trainer=trainer
        )

        train_loader, validation_loader, test_loader = data_module.predict_dataloader()
        y_pred, _ = model.predict(train_loader)
        accuracy_train = get_accuracy(y_train, y_pred)

        y_pred, _ = model.predict(validation_loader)
        accuracy_validation = get_accuracy(y_validation, y_pred)

        y_pred, _ = model.predict(test_loader)
        accuracy_test = get_accuracy(y_test, y_pred)

        write_to_logfile(pruning_threshold, variants_number, accuracy_train, accuracy_validation, accuracy_test)


if __name__ == '__main__':
    with open(LOG_FILE, 'w') as log_file:
        log_file.write('Accuracy on pruning threshold parameter for the model with centralized PCA\n')
    for pruning_threshold in [0.001, 0.002, 0.0035, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.035, 0.05, 0.1, 0.15, 0.2, 0.25]:
        run_experiment(pruning_threshold)
