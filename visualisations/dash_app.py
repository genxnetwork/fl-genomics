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
EXP_RUN_DICT = {'local_xgboost': ['1622d8f3b80a468890b15099910131e6', 'c2d6792921624997863a48c78f9ff6d6', '0cd45f5daebc49d89dd566366ddb4a94', 'c53c0641a96b4b6f85a0789018cb8d9e', 'd7d3ac5ab1ac44a785448c0bb1cf25c7', '048ee9d702924afa8375430616084a99', 'ea1607aecf014ad9aed1dce1f1020ae1', 'ab1e5a7e3bb84681b1589d40222f9c91', '46129dcc9de74113968938c6df49afc0', '702481a48bb24491b3c990e6f69c43a6', '7a374dfc94614096b67c4a32ad40aeb9', 'db16fc516a704db88d9aef8270edb2d4', '68f28d7f7b414506aa661d2e4ad25df4', '82203d64f1584d988318bc9b575a4421', 'f7bf582fe478440e878d3cfa1cce9ebe', '18e47b72322642e29a3c695b211c85e1', '46598602872f4e9c9fae71a13c9bcb51', '40b855b00a9d40bfb2d66b2e24a726b3', '814ca65bf7b7404480c6fe721c7c61cf', 'ce4e9a2983114510845084fa00488acd', '79edeb66ee7f46dab44fb153fc14de65', '31f6de19b6dd44f0b6e63962fd3947d6', '0beed056ccd347a295c4848de320aee8', '4d067368918749f18276e460a18ab79d', 'f9be155c5b394ee1b23011ebf09a846e', 'df18d7f1e60c48b4b1504d3cd42b02a5', 'e2b61a158e5d4b1997d602a43e4833d7', '02dd18a7ed4d4f03b8363fa55b9dae05', '7622ef1796c1442e80ac3b7da111cfb8', '85dd9a40f569475c899b49122e135168', '025dff91e6ef4abd91847b9168095fd7', 'c0e541dd408c463c99100bc3e5aa15a6', '30dc06c29e3e440fa6d6f8f002503f97', '530d06c9a9d34ce68e30051513fb2908', '063e5363f3554e9e81941ca08478c638', '10096987c8dc47798b2590417598ba09', '88147d59c5784e00ba21dccf0e2ddc7a', '185415f6e4bd42e7b22f8ace9e79b2e2', 'a9a3af72b72e4b1f836cc05bc4668d69', 'c2cc807f329541568aa814b14f9e26b6', '82ab44d7be1e447fb020202e6a685d5a', '6f72b16722fc4ec99bf40f798c899474', '022ecca65645445d8231d579309e5f6a', '48b181597e8246dc80e21d9093f199c6', 'e77da83a426f4829a25ef4a6e5d9726f', 'bbecc2f5742344879aa75a28ccc722ab', '5b0e73683244466789fd779668d37cfe', '16a9c9992359498aaedc77ffa6759fc8', '31e5c3eb536f417c97c5183dc2a51484', '9b4302e32dd04a8785145e6a0bd0ad8c', '8d97e939319b41cab590d0673ac58172', 'bc0516e42e61404395b1e659619b7ed2', '162f6b1137b542329b630cadb952e6e7', 'f3d1c067e75e4c26b6554749590f63ec', '0ed8d14d6184485db56f6642f12c328e', '869ebed86db54bf2a789e02870b10599', '612111f1978643e3b2eb457dd5281985', 'f8e6d2516f7a49b39136855c384e1a9c', 'f0d02f25ead44bf99927608602158d78', '252d1994cd564376b5a360b588538b62', '3e98ded3fe8141d5b5484c336286fcec', 'ed24d0fb59934d68a5693f7899320059', '66ab1e96461a46e882e640ab9e88096f', 'ed943cc7d81b4c8f85cd512f02dee750', 'c9e1afc4ff2e409b928f9dfa9bac2bd4', 'e2d9fcc818d44066b4d351ab7c6a0b02', '2cb0fb7339574367b23431939e775504', 'a1b9d5bc99c042e79762d98265898669', '48a2cdc71d2243b8a2f9a80027031d72', '076031e5d1c042ae9fffb88fe0896b60', '06824ed8ef9042cfb150b0fce8a8f579', '51b681489d5e45379f15a874b5817d00', '151a7d82cccb4693b19cf41eaa060797', 'ac0f0f51fb2c4d3099d579c3592331d5', 'ac326c1b5d37479f8a7a91f647d7c637'], 'local_lasso': ['974dfe37caae4b84b85c6826bf0f5368', 'e2bb2441cbfe45d1aac1f932d6e94504', 'd64458b7c60447138f602258c448878b', 'c2949203c7ba4cf0a1ffee0df8fcc65e', 'bc7ddb5d4bd2487fbeb8a87974a036eb', '5bcc96d917374869b2f58115c6cac986', '4e869e6b54764a9aac34d3f9a05e4d19', '1a2b1d79bc0443498cfff4127134da7f', '128938fd60284d1bb18517c92058a184', 'd0e843bf80824b87b9da000b6fed3c76', 'f00b876733f24e9f82fd089081407b42', 'f937e64ec4dc40c892d394e128547725', 'a0e9e43feb684f6f94b5f615b1502fb4', '0c8f14de962e4741ac5b367dc07bd430', 'c70a6b24e67a4b628257f0b17b903f4c', 'bfe8838445a94d2c9baf6e166d1aded4', '91cafe2224f040feb631cfd50bcb4cbc', '9d9f0f5f13a541d496f014170e5675b5', 'e2876dd0e81f44b690640f7758cdcba4', 'fe2dd1131f544b7d9166854617c21b38', 'ad311492d2be4e85906c53dd0ff69aac', '0c41951d9cc14ec2b125216a263d0371', '08201d1885864ec6b209efb648ae3c96', '4fcee7cd89c145f5b69863474388707e', 'e4d67156b48d4fc59beed921033ce12e', '20772a1374fb4d33a697f781eecc613a', '0af0697dda384b01a148d50ee91c8f13', 'cd7b69a344e44e2abf8a7518e7bbb263', '4203f60b0fc04e2497f6657d6e3817af', 'c6293c0ea139488fa95262da3e7ce936', '0c3566e81df840af87ae23685f45ef4b', '436f8766d05f47e091d346ea82b68661', '0d20a40df8684207a78d31313f5b8689', '48386a3581614740be282efd51e216c4', '7267a7888e8640c1a152afe33b503268', '3b872963be294ebeac5b9e5e55148ff4', '1cfab2edaf724cbf90909f9b47eba52c', 'df15a42cc2404c90b52e180a879e74f2', '6a476d284e1b41d8bd68d0f99febc440', '4023719f7d174f6b8e01e91230b03b89', '8a81e114a4b644658350029104252efb', 'a672615446334337ab597079bf38f30c', '7cab97e81843410da1169a50d1e6a740', '23c75a963b9c4f9eb81ede08c75bba72', 'cdc228eaa1444fe8ab371a8037cbe4e5', 'ae53bbd194e34cef8be6118238a65ade', 'b10bad0e5a6d42e6b486f0ecc017b788', '6b5fe0ea48774d81ba8fa49d69fa69c0', 'd96524b06ecf4df5bfcf45da3d68a1d0', '596cc5cd791d4eedb133009981e1ad2f', '60df49f8e7184e33b8cde0ffd0322df4', '263fd240d1284eae9fe1d45038a98d6b', '0bf3a9767731473a970834f881672f1b', 'd1938bb4ab7b43829500dbc4fe942ea1', '28ee5412627c4971ab2e92e98b5e0307', '3230d3882afc4a6bad89f5fbcb27c204', 'cccaff485bc7415988f78a6fe886d83c', '72e5752160df4d4391eca58e5f6c967f', '03f87b3f15114ed8b32e6e57d9ec508c', '0a676f66e1f148fc863692d32dab2631', '0b89e670ea4f4ccf917605f1aafa6309', 'af2f4ee7b1624f7e9149a984946fd1b4', 'f366b35aeb524050880bead2dacd4086', 'a721a9799e654b23a3e7a0bf2faa23cd', 'd15ddd6afde9455ea0c1522ab2390047', '457d6fa6fcbb4ad1b2fdf82e6e0a6e23', '90c7010bde2d4e40a87e1676413e971e', 'c219914d3a6841bda2ababcb936efa1d', '6b391593c6c1495f90421194f4cecbaf'], 'local-mlp-standing-height': ['a7c37ac1ffc645cdb5fca9f7dedba670', 'e199cbfa617048d6a0e90740bc820b5b', 'a0e703f96bca4126b71f7915d1f6a611', '0ffe5ab709b84da6b17a7f99b11f2e87', '4fd09f8d08f94b1c929910aecd482cdf', '846c98f66c784d08bf244d51ee5c1208', '482cc7b7afc24704a61d420e22c25643', 'b0037e47e5f645bcb94077b0247443fd', 'c45f561ae4ab4b51bcabdbe0a75482f8', '33d071ec73054472839f337b18c5f16b', '7fabfa6e7bde48c488854c4fed996080', '7fd6742446a84968b0b4c0e3a7fda785', '3f70a8c12bf64fb4b032caa7d75df8fd', '1c5bb44543df4badaaaf566d2a99617c', '084ff6d34d1d43e0b6abedba1f0119bc', 'e703f2b50ce4416286b0fc1f7aca274e', '244f1bbd451047ffa1c2d670e77f1656', '9da60377542a49c8b5353135bb6f1100', 'd38e8248863a44b89f6afe3f3d73971f', '92a46ad3a7cd41b9b603facc6c44284d', '279560423b7142959c62379ecf8c0d54', '0eb03848c22646ae9d1c029f0c01661f', '29f9c1e88fdb4cf4bd5095ad6103fe74', '98a93c90089f4233af177095401937ef', '7c202166dd10414eb3aabc80987e2180', '6db021c1f0d14b1982c93559952229d0', '6e03168f04ce488ba885d3ff1fed64cc', '158c330ec36745138c9d5fdb0114e7cd', 'f821d813522949749a971e86dc9f24dc', '262ccd9922774febb0bd68497b0d40e4', '4d4d90c9910b4c299e216f41eb69aaaf', '8b90941c9f694101a8c1137eb9cd3e9d', 'd08ccdc173bd49d38d81a416427cd9dc', 'f85ff165ce864ef88040bce7f125f835', '279af228f450447e9dd2d3edfc8daf23', '580b9d4bef1045c7ae7e19be8abf1c11', 'e82ef7f897f5453f80f77e7424e2da36', 'c264fea0d1154d49a5fce41e49c23657', '688084166c0942128c5f9e8fd3c7f160', '6b38a6d7c1d84d549903826f711c2295', '9e66935637a049d2a2afbd53b2b17772', 'dbf559d18b5b48129a795cd60f921254', 'a2c76a61c0374137915c0f2ef16ef6f5', '2612a43de4e24eb7a50fb585260ebefa', '25bce91b0ecb49a28f1a38f845cfb0fd', 'eb62116a49dc4b3dbadfa61112a1a29f', '3216ae351aae4ab282e0e723a248d868', 'c2eb5d3841df4057bef46fdce42c8f51', 'df02f5d185c24a2f92f05b4a2f928e6f', 'a989c70123a64f90918a2c2b6efd1421'], 'local-lassonet-standing-height': ['d2b42b6341374d56a83072d43eb9d183', 'c1e25498d923498d83fe4ddc83d7dd7e', 'e8dd4399b3e74261b2acc56bd3f0f29b', 'c77cd894009a4658b54934159b8df9f8', 'ad83a783eea64bcc9192a576836b0260', '406218519e2a47c4816fcc115888694d']}

# exp_run_dict = {exp_name: pd.read_csv(f'/home/dkolobok/Downloads/{exp_name}.csv')['Run ID'].tolist() for exp_name in EXP_RUN_DICT.keys()}


def mlflow_get_results_table(client):
    df = pd.concat([pd.read_csv(f'/home/dkolobok/Downloads/{exper_name}.csv') for exper_name in EXP_RUN_DICT.keys()])
    # run_dict = {exper_name: client.search_runs(experiment_ids=exper_ids) for exper_name, exper_ids in EXP_RUN_DICT.items()}
    # for exper_name in EXP_RUN_DICT.keys():
    #     client.search_runs(experiment_ids=EXP_RUN_DICT[exper_name])
    #     pass
    # client.search_runs(experiment_ids=["your_exprmnt_id"])
    return df[['name', 'node_index', 'snp_count', 'gwas_path', 'test_r2']].rename(columns={'name': 'model', 'node_index': 'num_samples', 'snp_count': 'num_snps'})

df = mlflow_get_results_table(client=MlflowClient())
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div([html.Button('Submit', id='submit_button', n_clicks=0)]),
    html.Div([
        html.Div([
            html.P('x'),
            dcc.Dropdown(
                df.columns,
                'num_snps',
                id='dd_x',
            )],
            style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            html.P('y'),
            dcc.Dropdown(
                df.columns,
                'test_r2',
                id='dd_y',
            )],
            style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            html.P('color'),
            dcc.Dropdown(
                df.columns,
                'num_samples',
                id='dd_color',
            )],
            style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            html.P('line_dash'),
            dcc.Dropdown(
                df.columns,
                'gwas_path',
                id='dd_line_dash',
            )],
            style={'width': '20%', 'display': 'inline-block'}),
        html.Div([
            html.P('filter by'),
            dcc.Dropdown(
                df.columns,
                'model',
                id='dd_filter_by',
            ),
            html.P('filter value'),
            dcc.Dropdown(
                df.model.unique(),
                'xgboost',
                id='dd_filter_value',
            )],
            style={'width': '20%', 'display': 'inline-block'}),
        ]),

    html.Div([
        dcc.Graph(id='results_comparison'),
    ], style={'display': 'inline-block', 'width': '100%'})
])


@app.callback(
    Output('dd_filter_value', 'options'),
    Input('dd_filter_by', 'value')
)
def update_filter_value(filter_by):
    return df[filter_by].unique().tolist()


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
