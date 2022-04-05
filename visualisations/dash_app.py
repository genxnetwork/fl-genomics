import os

import mlflow
import plotly.express as px
from dash.dependencies import Input, Output, State, ALL
from dash import html, dcc, Dash

from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

SELECTED_COLS = ['tags.phenotype', 'tags.model', 'tags.dataset', 'tags.snp_count',
                 'tags.not_full_WB_gwas', 'tags.sample_count', 'tags.ethnicity', 'metrics.test_r2']
GRAPH_ELEMENTS = ['x', 'y', 'color', 'line_dash', 'symbol']


def mlflow_get_results_table(client):
    exp_id = client.get_experiment_by_name("local-models").experiment_id
    df = mlflow.search_runs(experiment_ids=[exp_id])
    df[['tags.snp_count', 'tags.sample_count']] = df[['tags.snp_count', 'tags.sample_count']].astype(int)
    df['tags.ethnicity'] = df['tags.dataset'].str.split('_').str[0]
    df['tags.not_full_WB_gwas'] = 1 - df['tags.different_node_gwas'].astype(int)  # boolean NOT
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
                         id={
                             'type': 'dd_element',
                             'index': col_name,
                         }),
            style={'width': '33%', 'display': 'inline-block'}
        ),
        html.Div(
            dcc.Dropdown(df[col_name].unique().tolist() + ['All'],
                         'All',
                         id={
                             'type': 'dd_value',
                             'index': col_name,
                         }),
            style={'width': '33%', 'display': 'inline-block'}
        )
    ])
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
    State({'type': 'dd_element', 'index': ALL}, 'value'),
    State({'type': 'dd_value', 'index': ALL}, 'value'),
)
def update_graph(n_clicks, elements, values):
    if (n_clicks > 0) & ('x' in elements) & ('y' in elements):
        dfc = df.copy()
        line_args = {}
        for i, (element, value) in enumerate(zip(elements, values)):
            if element == 'filter':
                if value != 'All':
                    dfc = dfc[dfc.iloc[:, i] == value]
            else:
                line_args[element] = dfc.columns[i]
        fig = px.line(dfc.sort_values(dfc.columns[elements.index('x')]), **line_args, hover_data=df.columns)
        fig.update_xaxes(type='category')
        return fig
    return {}


if __name__ == '__main__':
    app.run_server(debug=True)
