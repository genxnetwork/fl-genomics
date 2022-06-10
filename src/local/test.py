import hydra
from omegaconf import OmegaConf
import mlflow
import numpy as np
from pytorch_lightning import Trainer
import torch
import numpy as np
from sklearn.metrics import r2_score
from local.experiment import LassoNetExperiment

# hydra.core.global_hydra.GlobalHydra().clear()
hydra.initialize(config_path="configs")
cfg = hydra.compose(config_name="test", overrides=['+model=lassonet'])

print(OmegaConf.to_yaml(cfg))

experiment = LassoNetExperiment(cfg)
experiment.load_sample_indices()
experiment.load_data()

train_node_index=1
loaded = mlflow.pytorch.load_model(f'models:/lassonet_region_split_node_{experiment.cfg.train_node_index}/1')
loaded.eval()

trainer = Trainer(gpus=0)
train_preds, val_preds, test_preds = trainer.predict(loaded,
                                                     [experiment.data_module.train_dataloader(),
                                                      experiment.data_module.val_dataloader(),
                                                      experiment.data_module.test_dataloader()])

train_preds = torch.cat(train_preds).numpy()
val_preds = torch.cat(val_preds).numpy()
test_preds = torch.cat(test_preds).numpy()

r2_val_list = [r2_score(experiment.y_val, val_preds[:, col]) for col in range(val_preds.shape[1])]
best_col = np.argmax(r2_val_list)
best_val_r2 = np.amax(r2_val_list)
best_train_r2 = r2_score(experiment.y_train, train_preds[:, best_col])
best_test_r2 = r2_score(experiment.y_test, test_preds[:, best_col])

mlflow.set_experiment(experiment.cfg.experiment.name)
with mlflow.start_run(tags={'test_node_index': str(experiment.cfg.node_index),
                            'train_node_index': str(experiment.cfg.train_node_index)}):
    print(f"Train r2: {best_train_r2}")
    mlflow.log_metric('train_r2', best_train_r2)
    print(f"Val r2: {best_val_r2}")
    mlflow.log_metric('val_r2', best_val_r2)
    print(f"Test r2: {best_test_r2}")
    mlflow.log_metric('test_r2', best_test_r2)