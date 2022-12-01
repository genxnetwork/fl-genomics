import os

import plotly.io as pio
pio.kaleido.scope.mathjax = None

import plotly.graph_objs as go
import plotly.express.colors as px_colors


class AccuracyOnPruningPlot:
    EMPTY_LINE = dict(color='rgba(0, 0, 0, 0)')

    def __init__(
            self,
            trace_colors=px_colors.qualitative.Plotly,
            band_colors=px_colors.qualitative.Plotly,
            cost_per_snp_mb=None
        ):

        self.fig = go.Figure()
        self.traces = []
        self.bands = []

        assert cost_per_snp_mb is not None
        self.cost_per_snp_mb = cost_per_snp_mb

        assert len(trace_colors) > 1 and len(band_colors) > 1
        self.centralized_trace_color = trace_colors[0]
        self.centralized_band_color = band_colors[0]
        self.federated_trace_color = trace_colors[1]
        self.federated_band_color = band_colors[1]

        self.centralized_dash_pool = ['solid', 'dashdot', 'dot']
        self.n_centralized_dash = 0

        self.centralized_pattern_pool = [None, 'x', '.']
        self.n_centralzied_pattern = 0

        self.federated_dash_pool = ['solid', 'dashdot', 'dot']
        self.n_federated_dash = 0

        self.federated_pattern_pool = [None, 'x', '.']
        self.n_federated_pattern = 0

        # ---
        self.fig.update_layout(
            xaxis=dict(title='Number of SNPs'),
            xaxis2=dict(title='Communication costs (GB)', anchor='free', overlaying='x', side='top', position=0.8),
            yaxis1=dict(title='Centralized model accuracy', domain=[0, 0.8])
        )

        self.fig.update_layout(title_text='MLP Accuracy for Centralized and Federated PCA')
        self.fig.update_layout(autosize=False, width=800, height=500)
        self.fig.update_layout(legend=dict(y=0.85))

        self.config = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'centralized-vs-federated-pca',
                'width': 800,
                'height': 500,
                'scale': 3
            }
        }

    @staticmethod
    def get_fillpattern(shape):
        if shape is None:
            return dict()
        else:
            return dict(shape=shape, size=4, solidity=0.4, fgcolor='white')

    def set_xrange(self, left, right):
        self.fig.update_layout(xaxis=dict(range=[left, right]))
        self.fig.update_layout(xaxis2=dict(range=[left, right]))

    def set_xticks(self, snps_tickvals, cost_tickvals, cost_ticktext):
        self.fig.update_layout(
            xaxis = dict(
                tickmode='array',
                tickvals=snps_tickvals
            )
        )

        self.fig.update_layout(
            xaxis2 = dict(
                tickmode='array',
                tickvals=cost_tickvals,
                ticktext=cost_ticktext
            )
        )

    def set_yrange(self, bottom, top):
        self.fig.update_yaxes(dict(range=[bottom, top]))

    def add_centralized_model_trace(self, snps, accuracy, name='Centralzied PCA (median)'):
        assert self.n_centralized_dash < len(self.centralized_dash_pool)
        dash = self.centralized_dash_pool[self.n_centralized_dash]
        self.n_centralized_dash += 1

        line = dict(color=self.centralized_trace_color, width=3, dash=dash)
        self.add_model_trace(snps, accuracy, line, name, legendrank=0)

    def add_centralized_model_band(self, snps, accuracy_lower, accuracy_upper, name='Centralized PCA (band)'):
        assert self.n_centralzied_pattern < len(self.centralized_pattern_pool)
        shape = self.centralized_pattern_pool[self.n_centralzied_pattern]
        self.n_centralzied_pattern += 1

        fillpattern = self.get_fillpattern(shape)
        self.add_model_band(
            snps,
            accuracy_lower,
            accuracy_upper,
            fillcolor=self.centralized_band_color,
            fillpattern=fillpattern,
            name=name,
            legendrank=2
        )

    def add_federated_model_trace(self, snps, accuracy, name='Federated PCA (median)'):
        assert self.n_federated_dash < len(self.federated_dash_pool)
        dash = self.federated_dash_pool[self.n_federated_dash]
        self.n_federated_dash += 1

        line = dict(color=self.federated_trace_color, width=3, dash=dash)
        self.add_model_trace(snps, accuracy, line, name, legendrank=1)

    def add_federated_model_band(self, snps, accuracy_lower, accuracy_upper, name='Federated PCA (band)'):
        assert self.n_federated_pattern < len(self.federated_pattern_pool)
        shape = self.federated_pattern_pool[self.n_federated_pattern]
        self.n_federated_pattern += 1

        fillpattern = self.get_fillpattern(shape)
        self.add_model_band(
            snps,
            accuracy_lower,
            accuracy_upper,
            fillcolor=self.federated_band_color,
            fillpattern=fillpattern,
            name=name,
            legendrank=3
        )

    def add_model_trace(self, snps, accuracy, line, name, legendrank):
        trace = go.Scatter(
            x=snps,
            y=accuracy,
            line=line,
            mode='markers+lines',
            name=name,
            legendrank=legendrank
        )

        self.traces.append(trace)

    def add_model_band(self, snps, accuracy_lower, accuracy_upper, fillcolor, fillpattern, name, legendrank):
        snps = snps + snps[::-1]
        accuracy = accuracy_upper + accuracy_lower[::-1]
        band = go.Scatter(
            x=snps,
            y=accuracy,
            line=AccuracyOnPruningPlot.EMPTY_LINE,
            fill='toself',
            fillcolor=fillcolor,
            fillpattern=fillpattern,
            hoverinfo='skip',
            name=name,
            legendrank=legendrank
        )

        self.bands.append(band)

    def set_selected_variants_number(self, number):
        trace = go.Scatter(
            x=[number, number],
            y=[0, 1],
            xaxis='x',
            yaxis='y1',
            line=dict(color='gray', dash='dot', width=3),
            mode='lines',
            legendrank=4,
            name='Selected SNPs number'
        )

        self.traces.append(trace)

    def fillup_figure(self):
        if len(self.fig.data) > 0:
            self.fig.data = []

        # Makes the 2nd xaxis visible. Be careful while moving this piece of code!
        empty_plot = go.Scatter(x=[], y=[], xaxis='x2', yaxis='y1')
        self.fig.add_trace(empty_plot)

        # Draw all the band first to drow lines on top of them
        for trace in self.bands:
            self.fig.add_trace(trace)

        for trace in self.traces:
            self.fig.add_trace(trace)

    def show(self):
        self.fillup_figure()
        self.fig.show(config=self.config)

    def export(self, filename):
        self.fillup_figure()
        _, ext = os.path.splitext(filename)
        format = ext[1:].lower()

        if format == 'png':
            pio.write_image(self.fig, filename, format='png', scale=3)
        elif format == 'pdf':
            pio.write_image(self.fig, filename, format='pdf')
        else:
            raise ValueError(f'Unsupported format: {format}')
