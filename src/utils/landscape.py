from nn.models import BaseNet
import numpy
import itertools
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import mlflow


def plot_loss_landscape(model: BaseNet, x: numpy.ndarray, y: numpy.ndarray, beta: numpy.ndarray):
        
        beta_space, Z = mse_on_beta_grid(x, y)
        # print(Z[90:, 20:30])
        fig = go.Figure()
        fig.add_trace(go.Contour(
                z=Z, # I don't know why we need T to work
                x=beta_space, # horizontal axis
                y=beta_space, # vertical axis,
                contours=dict(start=numpy.nanmin(Z), end=numpy.nanmax(Z), size=0.5)
        ))
        beta_history = numpy.array(model.beta_history).squeeze()
        # print(f'beta_history[0] shape is {self.model.beta_history[0].shape}')
        # print(f'beta_history shape is {beta_history.shape}')
        print(f'last betas: {beta_history[-1, :]}\ttrue betas: {beta}')
        # print(beta_history)
        print()
        add_beta_to_loss_landscape(fig, beta, beta_history, 'SGD')
        return fig


def mse_on_beta_grid(x: numpy.ndarray, y: numpy.ndarray, points_num: int = 100, beta_range=(-2, 2)):
    beta_space = numpy.linspace(beta_range[0], beta_range[1], num=points_num, endpoint=True)
    Z = numpy.zeros((points_num, points_num))
    for i, j in itertools.product(range(points_num), range(points_num)):
        beta_i, beta_j = beta_space[i], beta_space[j]
        y_pred = x.dot(numpy.array([beta_i, beta_j]).reshape(-1, 1))
        mse = mean_squared_error(y, y_pred[:, 0])
        # if abs(beta_i - self.beta[0]) < 0.01 and abs(beta_j - self.beta[1]) < 0.01:
        #     print(f'we have mse on contour plot: {mse:.5f} for {beta_i} and {beta_j}')
        Z[i, j] = mse

    print(f'local beta_grid shape is {Z.shape}')
    return beta_space, Z.T


def add_beta_to_loss_landscape(fig: go.Figure, true_beta: numpy.ndarray, beta_history: numpy.ndarray, name: str):
    fig.add_trace(go.Scatter(x=beta_history[:, 0], y=beta_history[:, 1], mode='markers+lines', name=name))
    fig.add_trace(go.Scatter(x=true_beta[0], y=true_beta[1], mode='markers', name=f'True beta {name}'))
    fig.update_layout(
        autosize=False,
        width=1536,
        height=1024,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))
