import logging
import os

import mlflow
import plotly.express as px
from dash.dependencies import Input, Output, State, ALL
from dash import html, dcc, Dash

from mlflow.tracking import MlflowClient

from utils.mlflow import mlflow_get_results_table, assemble_metric_history
from visualisations.dash_shared import DashApp

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(asctime)s %(message)s')
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

CURRENT_EXPERIMENT_NAME = "federated-tg-avg-epochs-8192-rounds-lr01"

SELECTED_TAGS = ['tags.phenotype',
 # 'tags.node_index',
 'tags.description',
 'tags.split']
# SELECTED_METRICS = [
#     # 'test_loss',
#  # 'test_accuracy',
#  # 'raw_loss',
#  'train_loss',
#  # 'lr',
#  'val_accuracy',
#  'val_loss',
#  'train_accuracy'
# ]
PER_EPOCH_METRICS = [
    # 'train_loss',
    'val_loss',
]
PER_ROUND_METRICS = [
    # 'train_accuracy',
    # 'val_accuracy',
]
# SELECTED_METRICS = [
#     # 'metrics.test_loss',
#  'metrics.test_accuracy',
#  # 'metrics.raw_loss',
#  # 'metrics.train_loss',
#  # 'metrics.lr',
#  'metrics.val_accuracy',
#  # 'metrics.val_loss',
#  'metrics.train_accuracy']
GRAPH_ELEMENTS = ['x', 'y', 'color', 'line_dash', 'symbol']

df = assemble_metric_history(client=MlflowClient(), experiment_name=CURRENT_EXPERIMENT_NAME,
                             selected_tags=SELECTED_TAGS, per_epoch_metrics=PER_EPOCH_METRICS, per_round_metrics=PER_ROUND_METRICS)
# df = mlflow_get_results_table(client=MlflowClient(), experiment_name=CURRENT_EXPERIMENT_NAME,
#                               selected_tags=SELECTED_TAGS, selected_metric=SELECTED_METRICS)

# app layout
app = Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
da = DashApp(df)
app.layout = da.create_layout()


# app callbacks
@app.callback(
    Output('results_comparison', 'figure'),
    Input('submit_button', 'n_clicks'),
    State({'type': 'dd_element', 'index': ALL}, 'value'),
    State({'type': 'dd_value', 'index': ALL}, 'value'),
)
def update_graph(n_clicks, elements, values):
    return da.update_graph(n_clicks=n_clicks, elements=elements, values=values, if_x_category=False)


if __name__ == '__main__':
    # run and open the link that appears in console in a browser
    # there assign filter+value or graph elements (x-axis, y-axis, color, etc.) to columns via dropdowns
    # then press submit
    app.run_server(debug=True)
