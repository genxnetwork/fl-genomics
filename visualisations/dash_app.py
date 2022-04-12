import os

import mlflow
import plotly.express as px
from dash.dependencies import Input, Output, State, ALL
from dash import html, dcc, Dash

from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

SELECTED_TAGS = ['tags.phenotype', 'tags.model', 'tags.dataset', 'tags.snp_count',
                 'tags.full_WB_gwas', 'tags.sample_count', 'tags.ethnicity']
SELECTED_METRICS = ['metrics.test_r2', 'metrics.val_r2', 'metrics.train_r2']
GRAPH_ELEMENTS = ['x', 'y', 'color', 'line_dash', 'symbol']


def mlflow_get_results_table(client):
    """ Retrieve local models from an mlflow client """
    exp_id = client.get_experiment_by_name("local-models").experiment_id
    df = mlflow.search_runs(experiment_ids=[exp_id])
    df[['tags.snp_count', 'tags.sample_count']] = df[['tags.snp_count', 'tags.sample_count']].astype(int)
    df['tags.ethnicity'] = df['tags.dataset'].str.split('_').str[0]
    df['tags.full_WB_gwas'] = df['tags.different_node_gwas'].astype(int)  # boolean NOT

    # melting dataframe to move from (metric1, metric2, ...) columns to (metric_name, metric_value) columns
    # for easier visualisation
    df = df.set_index(SELECTED_TAGS)[SELECTED_METRICS]
    if not df.index.is_unique:
        print('Warning! Runs with duplicated tags are found!')
    ser = df.stack()
    ser.index.names = ser.index.names[:-1] + ['metric_name']
    ser.name = 'metric_value'
    return ser.reset_index()


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
            dcc.Dropdown(df[col_name].dropna().unique()[:50].tolist() + ['All'],
                         'All',
                         id={
                             'type': 'dd_value',
                             'index': col_name,
                         }),
            style={'width': '33%', 'display': 'inline-block'}
        )
    ])
    block_list.append(block)

# app layout
app.layout = html.Div([
    html.Div([html.Button('Submit', id='submit_button', n_clicks=0)]),
    html.Div(block_list, style={'display': 'inline-block', 'width': '30%'}),
    html.Div([
        dcc.Graph(id='results_comparison'),
    ], style={'display': 'inline-block', 'width': '100%'})
])


# app callbacks
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
        fig = px.line(dfc.sort_values(dfc.columns[elements.index('x')]), **line_args, hover_data=df.columns,
                      markers=True)
        fig.update_xaxes(type='category')
        return fig
    return {}


if __name__ == '__main__':
    # run and open the link that appears in console in a browser
    # there assign filter+value or graph elements (x-axis, y-axis, color, etc.) to columns via dropdowns
    # then press submit
    app.run_server(debug=True)
