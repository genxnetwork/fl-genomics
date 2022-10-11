import os

import mlflow
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State, ALL
from dash import html, dcc, Dash

from mlflow.tracking import MlflowClient

from utils.mlflow import mlflow_get_results_table
from visualisations.dash_shared import DashApp

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

CURRENT_EXPERIMENT_NAME = "ac-split-local-models"

SELECTED_TAGS = ['tags.phenotype', 'tags.model', 'tags.dataset', 'tags.snp_count',
                 'tags.full_WB_gwas', 'tags.sample_count', 'tags.ethnicity',
                 'tags.fold_index']
SELECTED_METRICS = ['metrics.test_r2', 'metrics.val_r2', 'metrics.train_r2']

def ethnicity_from_dataset(df: pd.DataFrame):
    df['tags.ethnicity'] = df['tags.dataset'].str.split('_').str[0]
    return df

df = mlflow_get_results_table(client=MlflowClient(), experiment_name=CURRENT_EXPERIMENT_NAME,
                              selected_tags=SELECTED_TAGS, selected_metric=SELECTED_METRICS,
                              rename_cols={'tags.different_node_gwas': 'tags.full_WB_gwas'},
                              cols_to_int=['tags.snp_count', 'tags.sample_count', 'tags.full_WB_gwas'],
                              custom_transform=ethnicity_from_dataset)

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
    return da.update_graph(n_clicks=n_clicks, elements=elements, values=values)


if __name__ == '__main__':
    # run and open the link that appears in console in a browser
    # there assign filter+value or graph elements (x-axis, y-axis, color, etc.) to columns via dropdowns
    # then press submit
    app.run_server(debug=True)
