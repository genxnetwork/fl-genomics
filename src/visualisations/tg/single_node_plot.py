import os

import plotly.io as pio
# pio.kaleido.scope.mathjax = None

import plotly.graph_objs as go
import plotly.express.colors as px_colors

from plotly import subplots


class SingleNodePlot:
    def __init__(self, yaxis='Loss', title='Learning curve', color=px_colors.qualitative.Plotly[0]):
        self.fig = subplots.make_subplots(rows=1, cols=2)
        self.color = color

        self.fig.update_layout(
            xaxis = dict(title_text='Total epochs', nticks=5),
            xaxis2 = dict(title_text='Total epochs')
        )

        self.fig.update_yaxes(title_text=yaxis)
        self.fig.update_layout(title_text=title)
        self.fig.update_layout(autosize=False, width=900, height=400)

        self.config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'single-node-loss',
                'width': 900,
                'height': 400,
                'scale': 3
            }
        }

    def set_left_xrange(self, left, right):
        self.fig.update_layout(xaxis=dict(range=[left, right]))

    def set_right_xrange(self, left, right):
        self.fig.update_layout(xaxis2=dict(range=[left, right]))

    def set_left_yrange(self, bottom, top):
        self.fig.update_layout(yaxis=dict(range=[bottom, top]))

    def set_right_yrange(self, bottom, top):
        self.fig.update_layout(yaxis2=dict(range=[bottom, top]))

    def add_node_trace(self, epoch, loss):
        line = dict(color=self.color, width=2, dash='solid')
        plot = go.Scatter(x=epoch, y=loss, line=line, showlegend=False, mode='lines')

        self.fig.add_trace(plot, row=1, col=1)
        self.fig.add_trace(plot, row=1, col=2)

    def set_right_ticks(self, tickvals):
        self.fig.update_layout(
            xaxis2 = dict(
                tickmode='array',
                tickvals=tickvals,
                tickformat='5.0f'
            )
        )

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
