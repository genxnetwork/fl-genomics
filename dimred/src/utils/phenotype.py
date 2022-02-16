import pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from .split import Split


def adjust_in_place(train_covariates: pandas.DataFrame, train_phenotype: pandas.DataFrame,
                    val_covariates: pandas.DataFrame, val_phenotype: pandas.DataFrame,
                    test_covariates: pandas.DataFrame, test_phenotype: pandas.DataFrame):
    
    X_train = train_covariates.iloc[:, 2:].values
    y_train = train_phenotype.iloc[:, -1].values
    X_val = val_covariates.iloc[:, 2:].values
    y_val = val_phenotype.iloc[:, -1].values
    X_test = test_covariates.iloc[:, 2:].values
    y_test = test_phenotype.iloc[:, -1].values

    print(f'adjusting for {train_covariates.columns[2:]}')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred, y_val_pred, y_test_pred = lr.predict(X_train), lr.predict(X_val), lr.predict(X_test)
    res_train = y_train - y_train_pred
    res_val = y_val - y_val_pred
    res_test = y_test - y_test_pred
    
    train_r2, val_r2, test_r2 = r2_score(y_train, y_train_pred), r2_score(y_val, y_val_pred), r2_score(y_test, y_test_pred)
    print(f'R^2 train: {train_r2:.4f}\tval: {val_r2:.4f}\ttest: {test_r2:.4f}')
    print(train_phenotype.shape, y_train.shape, val_phenotype.shape, y_val.shape, test_phenotype.shape, y_test.shape)
    train_phenotype.iloc[:, -1] = res_train
    val_phenotype.iloc[:, -1] = res_val
    test_phenotype.iloc[:, -1] = res_test


def adjust_and_write(split: Split, node_index: int, fold_index: int):
    train_covariates = pandas.read_table(split.get_pca_cov_path(node_index, fold_index, 'train'))
    val_covariates = pandas.read_table(split.get_pca_cov_path(node_index, fold_index, 'val'))
    test_covariates = pandas.read_table(split.get_pca_cov_path(node_index, fold_index, 'test'))

    train_phenotype = pandas.read_table(split.get_phenotype_path(node_index, fold_index, 'train', adjusted=False))
    val_phenotype = pandas.read_table(split.get_phenotype_path(node_index, fold_index, 'val', adjusted=False))
    test_phenotype = pandas.read_table(split.get_phenotype_path(node_index, fold_index, 'test', adjusted=False))

    adjust_in_place(train_covariates, train_phenotype,
                    val_covariates, val_phenotype,
                    test_covariates, test_phenotype)
    
    train_phenotype.to_csv(split.get_phenotype_path(node_index, fold_index, 'train', adjusted=True), sep='\t', index=False)
    val_phenotype.to_csv(split.get_phenotype_path(node_index, fold_index, 'val', adjusted=True), sep='\t', index=False)
    test_phenotype.to_csv(split.get_phenotype_path(node_index, fold_index, 'test', adjusted=True), sep='\t', index=False)
