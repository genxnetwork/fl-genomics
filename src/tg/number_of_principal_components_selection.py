import numpy as np

from torch.nn.functional import cross_entropy
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from nn.models import MLPClassifier
from utils.metrics import get_accuracy
from configs.split_config import FOLDS_NUMBER
from tg.data_provider import DataProvider


LOG_FILE = '/home/genxadmin/accuracy.log'


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


def log(num_components, accuracy_train, accuracy_validation, accuracy_test):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f'{num_components} {accuracy_train} {accuracy_validation} {accuracy_test}\n')


def run_experiment(num_components):
    """
    Runs training using the specified number of principal componenets and logs accuracy metrics
    to the file.
    """

    data_provider = DataProvider(
        '/mnt/genx-bio-share/TG/data/chip/superpop_split/pca',
        '/mnt/genx-bio-share/TG/data/chip/superpop_split/only_phenotypes/ancestry',
        num_components=num_components
    )

    for fold in range(FOLDS_NUMBER):
        _, y_train = data_provider.load_train_data('ALL', fold)
        _, y_validation = data_provider.load_validation_data('ALL', fold)
        _, y_test = data_provider.load_test_data('ALL', fold)
        data_module = data_provider.create_data_module('ALL', fold)

        model = get_model(num_classes=len(np.unique(y_train)), num_features=num_components)

        trainer = get_trainer()
        trainer.fit(model, data_module)

        model = get_model(num_classes=len(np.unique(y_train)), num_features=num_components, trainer=trainer)

        train_loader, validation_loader, test_loader = data_module.predict_dataloader()
        y_pred, _ = model.predict(train_loader)
        accuracy_train = get_accuracy(y_train, y_pred)

        y_pred, _ = model.predict(validation_loader)
        accuracy_validation = get_accuracy(y_validation, y_pred)

        y_pred, _ = model.predict(test_loader)
        accuracy_test = get_accuracy(y_test, y_pred)

        log(num_components, accuracy_train, accuracy_validation, accuracy_test)


if __name__ == '__main__':
    with open(LOG_FILE, 'w') as log_file:
        log_file.write('Accuracy for different number of principal components\n')
    for num_components in range(20, 0, -1):
        run_experiment(num_components)
