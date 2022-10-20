import os

import mlflow
import plotly.express as px
from dash.dependencies import Input, Output, State, ALL
from dash import html, dcc, Dash

from mlflow.tracking import MlflowClient

from utils.mlflow import mlflow_get_results_table

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

GRAPH_ELEMENTS = ['x', 'y', 'color', 'line_dash', 'symbol', 'errorbar_0.8']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

class DashApp(object):
    def __init__(self, df):
        self.df = df

    def create_html_blocks(self):
        # create html blocks for columns
        block_list = []
        for col_name in self.df.columns:
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
                    dcc.Dropdown(self.df[col_name].dropna().unique()[:50].tolist() + ['All'],
                                 'All',
                                 id={
                                     'type': 'dd_value',
                                     'index': col_name,
                                 }),
                    style={'width': '33%', 'display': 'inline-block'}
                )
            ])
            block_list.append(block)
        return block_list

    def create_layout(self):
        # app layout
        return html.Div([
            html.Div([html.Button('Submit', id='submit_button', n_clicks=0)]),
            html.Div(self.create_html_blocks(), style={'display': 'inline-block', 'width': '30%'}),
            html.Div([
                dcc.Graph(id='results_comparison'),
            ], style={'display': 'inline-block', 'width': '100%'})
        ])

    def update_graph(self, n_clicks, elements, values, if_x_category=True):
        if (n_clicks > 0) & ('x' in elements) & ('y' in elements):
            dfc = self.df.copy()
            line_args = {}
            original_columns = dfc.columns.copy()
            for i, (element, value) in enumerate(zip(elements, values)):
                if element == 'filter':
                    if value != 'All':
                        dfc = dfc[dfc[original_columns[i]] == value]
                elif element == 'errorbar_0.8':
                    dfc = dfc.groupby([x for x in dfc.columns if (x != 'tags.fold_index') & (x != 'metric_value')])['metric_value'].quantile([0.1, 0.5, 0.9]).unstack()
                    dfc['metric_value'] = dfc[0.5]
                    dfc['error_lower'] = dfc[0.5] - dfc[0.1]
                    dfc['error_upper'] = dfc[0.9] - dfc[0.5]
                    dfc = dfc.reset_index()
                    line_args['error_y'] = 'error_upper'
                    line_args['error_y_minus'] = 'error_lower'
                    pass
                else:
                    line_args[element] = original_columns[i]
            fig = px.line(dfc.sort_values(dfc.columns[elements.index('x')]), **line_args, hover_data=dfc.columns,
                          markers=True)
            if if_x_category:
                fig.update_xaxes(type='category')
            return fig
        return {}


if __name__ == '__main__':
    # run and open the link that appears in console in a browser
    # there assign filter+value or graph elements (x-axis, y-axis, color, etc.) to columns via dropdowns
    # then press submit
    app.run_server(debug=True)
