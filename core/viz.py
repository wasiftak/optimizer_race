# path: core/viz.py
import numpy as np
import plotly.graph_objects as go
import torch


def _grid(terrain, n=200):
    xmin, xmax, ymin, ymax = terrain.bounds
    xs = torch.linspace(xmin, xmax, n)
    ys = torch.linspace(ymin, ymax, n)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    Z = terrain.fn(torch.stack([X, Y]))
    # log1p compresses huge ranges (banana) so the contours stay readable
    Z = torch.log1p(torch.clamp(Z - terrain.fmin, min=0.0))
    return xs.numpy(), ys.numpy(), Z.numpy()


def contour_base(terrain):
    """Contour map of the terrain with a star at the global minimum."""
    xs, ys, Z = _grid(terrain)
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, colorscale='Viridis', showscale=False,
        contours=dict(coloring='heatmap'), opacity=0.85,
        hoverinfo='skip', name='loss'))
    fig.add_trace(go.Scatter(
        x=[terrain.minimum[0]], y=[terrain.minimum[1]], mode='markers',
        marker=dict(symbol='star', size=18, color='gold',
                    line=dict(color='black', width=1)),
        name='Finish (minimum)'))
    xmin, xmax, ymin, ymax = terrain.bounds
    fig.update_layout(
        xaxis=dict(range=[xmin, xmax], title='x (weight 1)'),
        yaxis=dict(range=[ymin, ymax], title='y (weight 2)'),
        height=560, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', y=1.08))
    return fig


def paths_figure(terrain, logs, colors):
    """Full path of every racer. colors: dict name -> hex colour."""
    fig = contour_base(terrain)
    xmin, xmax, ymin, ymax = terrain.bounds
    end_symbol = {'finished': 'star', 'crashed': 'x'}
    for log in logs:
        c = colors[log.name]
        # clamp so a diverging racer does not blow up the axes
        xs = np.clip(log.x, xmin, xmax)
        ys = np.clip(log.y, ymin, ymax)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines',
                                 line=dict(color=c, width=2.5), name=log.name))
        fig.add_trace(go.Scatter(
            x=[xs[0]], y=[ys[0]], mode='markers', showlegend=False,
            marker=dict(symbol='circle', size=11, color=c,
                        line=dict(color='white', width=2)),
            hoverinfo='skip'))
        sym = end_symbol.get(log.status, 'diamond')
        fig.add_trace(go.Scatter(
            x=[xs[-1]], y=[ys[-1]], mode='markers', showlegend=False,
            marker=dict(symbol=sym, size=16, color=c,
                        line=dict(color='white', width=2)),
            hovertext=f'{log.name}: {log.status}, step {log.steps[-1]}',
            hoverinfo='text'))
    return fig