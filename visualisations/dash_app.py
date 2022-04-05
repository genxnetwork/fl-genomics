import os

import pandas as pd
import dash
import mlflow
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash import html, dcc, Dash
# import dash_pivottable

# from data import data
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

# EXP_RUN_DICT = {
#     'local_xgboost': [],
#     'local_lasso': [],
#     'local-mlp-standing-height': [],
#     'local-lassonet-standing-height': []
# }
SELECTED_COLS = ['tags.phenotype', 'tags.model', 'tags.dataset', 'tags.snp_count', 'tags.mlflow.source.name',
                 'tags.different_node_gwas', 'metrics.test_r2']
GRAPH_ELEMENTS = ['x', 'y', 'color', 'dash_line', 'shape']
# exp_run_dict = {exp_name: pd.read_csv(f'/home/dkolobok/Downloads/{exp_name}.csv')['Run ID'].tolist() for exp_name in EXP_RUN_DICT.keys()}


def mlflow_get_results_table(client):
    exp_id = client.get_experiment_by_name("local-models").experiment_id
    df = mlflow.search_runs(experiment_ids=[exp_id])
    return df[SELECTED_COLS]

df = mlflow_get_results_table(client=MlflowClient())
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)


# create html blocks for columns
block_list = []
for col_name in df.columns:
    block = html.Div([
        html.Div(html.P(col_name), style={'width': '33%', 'display': 'inline-block'}),
        html.Div(
            dcc.Dropdown(GRAPH_ELEMENTS + ['filter'],
                         'filter',
                         id=f'dd_{col_name}_element'),
            style={'width': '33%', 'display': 'inline-block'}
        ),
        html.Div(
            dcc.Dropdown(df[col_name].unique(),
                         df[col_name].iloc[0],
                         id=f'dd_{col_name}_value'),
            style={'width': '33%', 'display': 'inline-block'}
        )
    ])


    # @app.callback(
    #     Output(f'dd_{col_name}_value', 'disabled'),
    #     Input(f'dd_{col_name}_element', 'value')
    # )
    # def update_filter_value(filter_by):
    #     return filter_by != 'filter'

    block_list.append(block)

app.layout = html.Div([
    html.Div([html.Button('Submit', id='submit_button', n_clicks=0)]),
    html.Div(block_list, style={'display': 'inline-block', 'width': '30%'}),
    html.Div([
        dcc.Graph(id='results_comparison'),
    ], style={'display': 'inline-block', 'width': '100%'})
])




@app.callback(
    Output('results_comparison', 'figure'),
    Input('submit_button', 'n_clicks'),
    State('dd_x', 'value'),
    State('dd_y', 'value'),
    State('dd_color', 'value'),
    State('dd_line_dash', 'value'),
    State('dd_filter_by', 'value'),
    State('dd_filter_value', 'value'),
)
def update_graph(n_clicks, x_col, y_col, color_col, line_dash_col,
                 filter_by_col, filter_by_value):
    if n_clicks > 0:
        fig = px.line(df[df[filter_by_col] == filter_by_value].sort_values(x_col), x=x_col, y=y_col, color=color_col, line_dash=line_dash_col)
        fig.update_xaxes(type='category')
        return fig
    return {}


if __name__ == '__main__':
    app.run_server(debug=True)


# app = dash.Dash(__name__)
# app.title = 'My Dash example'
#
# app.layout = html.Div([
#     dash_pivottable.PivotTable(
#         id='table',
#         data=data,
#         cols=['Day of Week'],
#         colOrder="key_a_to_z",
#         rows=['Party Size'],
#         rowOrder="key_a_to_z",
#         rendererName="Grouped Column Chart",
#         aggregatorName="Average",
#         vals=["Total Bill"],
#         valueFilter={'Day of Week': {'Thursday': False}}
#     ),
#     html.Div(
#         id='output'
#     )
# ])
#
#
# @app.callback(Output('output', 'children'),
#               [Input('table', 'cols'),
#                Input('table', 'rows'),
#                Input('table', 'rowOrder'),
#                Input('table', 'colOrder'),
#                Input('table', 'aggregatorName'),
#                Input('table', 'rendererName')])
# def display_props(cols, rows, row_order, col_order, aggregator, renderer):
#     return [
#         html.P(str(cols), id='columns'),
#         html.P(str(rows), id='rows'),
#         html.P(str(row_order), id='row_order'),
#         html.P(str(col_order), id='col_order'),
#         html.P(str(aggregator), id='aggregator'),
#         html.P(str(renderer), id='renderer'),
#     ]
