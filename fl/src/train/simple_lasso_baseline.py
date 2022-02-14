from statistics import mean
from omegaconf import DictConfig, OmegaConf
import hydra

from datasets.memory import load_from_pgen, load_phenotype
from datasets.lightning import DataModule

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score


@hydra.main(config_path='../configs/client', config_name='default')
def main(cfg: DictConfig):

    X_train = load_from_pgen(cfg.data.genotype.train, cfg.data.gwas, None, missing=cfg.experiment.missing) # load all snps
    X_val = load_from_pgen(cfg.data.genotype.val, cfg.data.gwas, None, missing=cfg.experiment.missing) # load all snps
    print('Genotype data loaded')
    print(f'We have {X_train.shape[1]} snps, {X_train.shape[0]} train samples and {X_val.shape[0]} val samples')

    y_train, y_val = load_phenotype(cfg.data.phenotype.train), load_phenotype(cfg.data.phenotype.val)
    print(f'We have {y_train.shape[0]} train phenotypes and {y_val.shape[0]} val phenotypes')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print(y_train.mean(), y_val.mean())
    lasso = LassoCV(max_iter=2000)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_val_pred = lasso.predict(X_val)
    
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    print(f'Train R^2 is {train_r2:.4f}, val R^2 is {val_r2:.4f}')
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)

    print(f'Train MSE is {train_mse:.4f}, Val MSE is {val_mse:.4f}')


if __name__ == '__main__':
    main()