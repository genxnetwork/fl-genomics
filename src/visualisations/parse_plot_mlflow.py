import os

import mlflow
import pandas as pd
import plotly.express as px
from mlflow.tracking import MlflowClient

# access the tracking server using environmental variables
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

class ParsePlotMLFlow(object):
    """ Parses and plots different metrics from mlflow tracking server """
    def __init__(self):
        self.client = MlflowClient()

    def parse_epochs(self, run_id: str, metrics_list: list) -> pd.DataFrame:
        """ Loads metrics from @metrics_list of the run @run_id and parse them into a single dataframe with
            index being a union of steps and column being metrics
        """
        df_list = []
        for metric in metrics_list:
            data = self.client.get_metric_history(run_id=run_id, key=metric)
            metric_dict = {data[i].step: data[i].value for i in range(len(data))}
            df_list.append(pd.DataFrame({metric: metric_dict.values()}, index=metric_dict.keys()))
        return pd.concat(df_list, axis=1)

    @staticmethod
    def visualise_epochs(df: pd.DataFrame, output_fn: str) -> None:
        """ Visualises metrics (y-axis) vs epochs (x-axis). Dataframe @df has metrics as columns and epochs as index """
        df = df.stack().reset_index()
        df.columns = ['epoch', 'metric', 'value']
        fig = px.line(df, x='epoch', y='value', color='metric')
        fig.write_html(output_fn)


if __name__ == '__main__':
    ppmfl = ParsePlotMLFlow()
    df = ppmfl.parse_epochs(run_id='388d2f1daf1a4eb7a72d11eb5de9429f', metrics_list=['val_loss'])
    ppmfl.visualise_epochs(df=df, output_fn='/home/dkolobok/Downloads/epochs.html')
