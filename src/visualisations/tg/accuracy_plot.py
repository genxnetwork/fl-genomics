import os

import plotly.io as pio
pio.kaleido.scope.mathjax = None

import plotly.graph_objs as go
import plotly.express.colors as px_colors

from plotly import subplots


class AccuracyPlot:
    def __init__(self, yaxis='Accuracy', colors=px_colors.qualitative.Plotly, cost_per_round_mb=None):
        self.fig = subplots.make_subplots(rows=1, cols=2)
        self.colors = colors
        assert cost_per_round_mb is not None
        self.cost_per_round_mb = cost_per_round_mb
        self.centralized_model_color = self.colors[0]
        self.federated_model_colors = self.colors[1:]
        self.n_federated_model_traces = 0 # controls color palette

        self.fig = subplots.make_subplots(rows=1, cols=2)
        self.fig.update_layout(
            xaxis = dict(title_text='Total epochs'),
            xaxis2 = dict(title_text='Total rounds'),
            xaxis3=dict(title='Communication costs (GB)', anchor='free', overlaying='x2', side='top', position=0.8),
            yaxis1=dict(domain=[0, 0.8]), yaxis2=dict(domain=[0, 0.8])
        )

        # Makes the 2nd xaxis visible. Be careful while moving this piece of code!
        empty_plot = go.Scatter(x=[], y=[], xaxis='x3', yaxis='y2')
        self.fig.add_trace(empty_plot)

        self.fig.update_yaxes(title_text=yaxis)
        self.fig.update_layout(title_text='Learning curves')
        self.fig.update_layout(autosize=False, width=900, height=425)
        self.fig.update_layout(legend=dict(y=0.85))

        self.config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'accuracy',
                'width': 900,
                'height': 425,
                'scale': 3
            }
        }

    def set_left_xrange(self, left, right):
        self.fig.update_layout(xaxis=dict(range=[left, right]))

    def set_right_xrange(self, left, right):
        self.fig.update_layout(xaxis2=dict(range=[left, right]))
        self.fig.update_layout(xaxis3=dict(
            range=[
                left * self.cost_per_round_mb,
                right * self.cost_per_round_mb
            ]
        ))

    def set_yrange(self, bottom, top):
        self.fig.update_layout(
            yaxis1=dict(range=[bottom, top]),
            yaxis2=dict(range=[bottom, top])
        )

    def add_centralized_model_trace(self, epoch, round, cost, accuracy, name='Centralized'):
        """
        Should we plot centralized model somehow?
        """
        line = dict(color=self.centralized_model_color, width=3)
        self.add_model_trace(epoch, round, cost, accuracy, name, line)

    def add_federated_model_trace(self, epoch, round, accuracy, name=None):
        assert self.n_federated_model_traces < len(self.federated_model_colors)
        line = dict(color=self.federated_model_colors[self.n_federated_model_traces], width=2)
        self.add_model_trace(epoch, round, accuracy, name, line)
        self.n_federated_model_traces += 1

    def add_model_trace(self, epoch, round, accuracy, name, line):
        left_plot = go.Scatter(
            x=epoch,
            y=accuracy,
            line=line,
            mode='lines',
            name=name,
            legendrank=0
        )

        right_plot = go.Scatter(
            x=round,
            y=accuracy,
            line=line,
            showlegend=False,
            mode='lines',
            name=name,
            legendrank=0
        )

        self.fig.add_trace(left_plot, row=1, col=1)
        self.fig.add_trace(right_plot, row=1, col=2)

    def set_right_ticks(self, round_tickvals, cost_tickvals, cost_ticktext):
        self.fig.update_layout(
            xaxis2 = dict(
                tickmode='array',
                tickvals=round_tickvals
            )
        )

        self.fig.update_layout(
            xaxis3 = dict(
                tickmode='array',
                tickvals=cost_tickvals,
                ticktext=cost_ticktext
            )
        )

    def set_pca_costs(self, cost):
        trace = go.Scatter(
            x=[cost, cost],
            y=[0, 1],
            xaxis='x3',
            yaxis='y2',
            line=dict(color='gray', dash='dot', width=3),
            mode='lines',
            legendrank=2,
            name='PCA costs'
        )

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
