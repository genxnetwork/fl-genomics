import pytest
from nn.utils import RegFederatedMetrics, LassoNetRegMetrics, RegLoaderMetrics


def test_RegFederatedMetrics_reduce():
    client1 = LassoNetRegMetrics(
        [RegLoaderMetrics('train', 0.5, 0.5, 1, 1000), RegLoaderMetrics('train', 0.25, 0.75, 1, 1000)], 
        [RegLoaderMetrics('val', 0.1, 0.9, 1, 1000), RegLoaderMetrics('val', 0.2, 0.8, 1, 1000)],       
        epoch=1
    )
    client2 = LassoNetRegMetrics(
        [RegLoaderMetrics('train', 0.75, 0.25, 1, 1000), RegLoaderMetrics('train', 0.5, 0.5, 1, 1000)], 
        [RegLoaderMetrics('val', 0.3, 0.7, 1, 1000), RegLoaderMetrics('val', 0.4, 0.6, 1, 1000)],       
        epoch=1
    )
    fed_metrics = RegFederatedMetrics([client1, client2], 1)
    red_metrics = fed_metrics.reduce('lassonet_best')
    assert red_metrics.best_col == 0
    assert red_metrics.train[0].loss == 0.625
    assert red_metrics.val[0].loss == 0.2
    assert red_metrics.val_loss == 0.2
    final_metrics = red_metrics.reduce('lassonet_best')
    assert final_metrics.val.r2 == 0.8
    assert final_metrics.val_loss == 0.2


def test_RegFederatedMetrics_reduce_unbalanced():
    client1 = LassoNetRegMetrics(
        [RegLoaderMetrics('train', 0.5, 0.5, 1, 1000), RegLoaderMetrics('train', 0.25, 0.75, 1, 1000)], 
        [RegLoaderMetrics('val', 0.1, 0.9, 1, 1000), RegLoaderMetrics('val', 0.2, 0.8, 1, 1000)],       
        epoch=1
    )
    client2 = LassoNetRegMetrics(
        [RegLoaderMetrics('train', 0.75, 0.25, 1, 2000), RegLoaderMetrics('train', 0.5, 0.5, 1, 1000)], 
        [RegLoaderMetrics('val', 0.3, 0.7, 1, 2000), RegLoaderMetrics('val', 0.4, 0.6, 1, 1000)],       
        epoch=1
    )
    fed_metrics = RegFederatedMetrics([client1, client2], 1)
    red_metrics = fed_metrics.reduce('lassonet_best')
    assert red_metrics.best_col == 0
    assert 0.66 < red_metrics.train[0].loss < 0.67
    assert 0.23 < red_metrics.val[0].loss < 0.24
    assert 0.23 < red_metrics.val_loss < 0.24
    final_metrics = red_metrics.reduce('lassonet_best')
    assert 0.76 < final_metrics.val.r2 < 0.77
    assert 0.23 < final_metrics.val_loss < 0.24