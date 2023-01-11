import os

import plotly.io as pio
pio.kaleido.scope.mathjax = None

import plotly.graph_objs as go
import plotly.express.colors as px_colors


class LossPlot:
    """
    Shows dependence of loss on the total number of epochs for different strategies.
    The plot visualizes convergence of the training process. Color palette is cotrolled
    by the `colors` parameter of the constructor.
    """

    def __init__(self, yaxis='Loss', colors=px_colors.qualitative.Plotly):
        self.fig = go.Figure()
        self.colors = colors
        self.centralized_model_color = self.colors[0]
        self.federated_model_colors = self.colors[1:]
        self.n_federated_model_traces = 0 # controls color palette

        self.fig.update_xaxes(title_text='Total epochs')
        self.fig.update_yaxes(title_text=yaxis)
        self.fig.update_layout(title_text='Learning curves')

        self.fig.update_layout(autosize=False, width=600, height=400)
        self.fig.update_layout(showlegend=True)

        self.config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'loss',
                'width': 600,
                'height': 400,
                'scale': 3
            }
        }

    def set_xrange(self, left, right):
        self.fig.update_xaxes(range=[left, right])

    def set_yrange(self, bottom, top):
        self.fig.update_yaxes(range=[bottom, top])

    def add_centralized_model_trace(self, epoch, loss, name='Centralized'):
        line = dict(color=self.centralized_model_color, width=3)
        self.add_model_trace(epoch, loss, name, line)

    def add_federated_model_trace(self, epoch, loss, name=None):
        assert self.n_federated_model_traces < len(self.federated_model_colors)
        line = dict(color=self.federated_model_colors[self.n_federated_model_traces], width=2)
        self.add_model_trace(epoch, loss, name, line)
        self.n_federated_model_traces += 1

    def add_model_trace(self, epoch, loss, name, line):
        trace = go.Scatter(x=epoch, y=loss, line=line, mode='lines', name=name)
        self.fig.add_trace(trace)

    def show(self):
        self.fig.show(config=self.config)

    def export(self, filename):
        _, ext = os.path.splitext(filename)
        format = ext[1:].lower()

        if format == 'png':
            pio.write_image(self.fig, filename, format='png', scale=3)
        elif format == 'pdf':
            pio.write_image(self.fig, filename, format='pdf')
        else:
            raise ValueError(f'Unsupported format: {format}')
