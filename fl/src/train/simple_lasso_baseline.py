from omegaconf import DictConfig, OmegaConf
import hydra

from datasets.memory import load_from_pgen, load_phenotype
from datasets.lightning import DataModule

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


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
    lasso = LassoCV()
    lasso.fit(X_train, y_train)
    train_r2 = lasso.score(X_train, y_train)
    val_r2 = lasso.score(X_val, y_val)
    print(f'Train R^2 is {train_r2:.4f}, val R^2 is {val_r2:.4f}')



if __name__ == '__main__':
    main()