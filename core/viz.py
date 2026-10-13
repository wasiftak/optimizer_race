# path: core/viz.py
import math
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


def contour_base(terrain, zone=None):
    """Contour map with a star at the minimum and an optional danger-zone rectangle."""
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
    if zone is not None:
        x0, x1, y0, y1 = zone
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=y0, y1=y1, layer='above',
                      line=dict(color='#e63946', width=2, dash='dash'),
                      fillcolor='rgba(230,57,70,0.18)')
        fig.add_annotation(x=x0, y=y1, text='Danger zone', showarrow=False,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='#e63946', size=12))
    return fig


def paths_figure(terrain, logs, colors):
    """Full path of every racer (static). colors: dict name -> hex colour."""
    fig = contour_base(terrain)
    xmin, xmax, ymin, ymax = terrain.bounds
    end_symbol = {'finished': 'star', 'crashed': 'x'}
    for log in logs:
        c = colors[log.name]
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


def _plot_path(terrain, log):
    """Plottable path. A crashed racer is cut at its last in-bounds position."""
    xmin, xmax, ymin, ymax = terrain.bounds
    x = np.array(log.x, dtype=float)
    y = np.array(log.y, dtype=float)
    ok = (np.isfinite(x) & np.isfinite(y) & (x >= xmin) & (x <= xmax)
          & (y >= ymin) & (y <= ymax))
    if log.status == 'crashed':
        idx = np.where(ok)[0]
        last = int(idx[-1]) if len(idx) else 0
        x, y = x[:last + 1], y[:last + 1]
    else:
        x, y = np.clip(x, xmin, xmax), np.clip(y, ymin, ymax)
    return x.tolist(), y.tolist()


def _car_marker(color, symbol):
    if symbol == 'x':
        return dict(symbol='x', size=18, color=color, line=dict(color=color, width=3))
    return dict(symbol=symbol, size=16, color=color, line=dict(color='white', width=2))


def animated_race_figure(terrain, logs, colors, stride=2, frame_ms=60, max_frames=150,
                         zone=None):
    """All racers move together. Plays in the browser via Plotly frames."""
    fig = contour_base(terrain, zone)
    base = len(fig.data)
    paths = [_plot_path(terrain, l) for l in logs]
    n = len(logs)

    for l, (xs, ys) in zip(logs, paths):          # trail traces
        fig.add_trace(go.Scatter(x=xs[:1], y=ys[:1], mode='lines',
                                 line=dict(color=colors[l.name], width=2.5),
                                 name=l.name))
    for l, (xs, ys) in zip(logs, paths):          # car traces
        fig.add_trace(go.Scatter(x=xs[:1], y=ys[:1], mode='markers',
                                 marker=_car_marker(colors[l.name], 'diamond'),
                                 showlegend=False, hoverinfo='skip'))

    T = max(len(xs) for xs, _ in paths)
    stride = max(1, int(stride), math.ceil(T / max_frames))
    ks = list(range(0, T, stride))
    if ks[-1] != T - 1:
        ks.append(T - 1)

    end_symbol = {'finished': 'star', 'crashed': 'x'}
    frames = []
    for k in ks:
        data = []
        for (xs, ys) in paths:                    # trails up to frame k
            i = min(k, len(xs) - 1)
            data.append(go.Scatter(x=xs[:i + 1], y=ys[:i + 1]))
        for l, (xs, ys) in zip(logs, paths):      # cars at frame k
            i = min(k, len(xs) - 1)
            sym = end_symbol.get(l.status, 'diamond') if i == len(xs) - 1 else 'diamond'
            data.append(go.Scatter(x=[xs[i]], y=[ys[i]],
                                   marker=_car_marker(colors[l.name], sym)))
        frames.append(go.Frame(data=data, traces=list(range(base, base + 2 * n)),
                               name=str(k)))
    fig.frames = frames

    # every step gets a label so "Step: N" always shows; the tick labels are hidden
    steps = []
    for k in ks:
        steps.append(dict(
            method='animate', label=str(k),
            args=[[str(k)], dict(mode='immediate',
                                 frame=dict(duration=0, redraw=False),
                                 transition=dict(duration=0))]))
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=90),
        updatemenus=[dict(
            type='buttons', direction='left', x=0.0, y=-0.1,
            xanchor='left', yanchor='top', showactive=False,
            buttons=[
                dict(label='Play', method='animate',
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=False),
                                      fromcurrent=True, mode='immediate',
                                      transition=dict(duration=0))]),
                dict(label='Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate',
                                        transition=dict(duration=0))]),
            ])],
        sliders=[dict(active=0, x=0.18, len=0.82, y=-0.08, yanchor='top',
                      currentvalue=dict(prefix='Step: ', font=dict(size=14)),
                      font=dict(color='rgba(0,0,0,0)'),
                      steps=steps)])
    return fig


def surface_figure(terrain, logs, colors, n=120):
    """3D surface (log-compressed loss) with each racer's path drawn on top."""
    xs, ys, Z = _grid(terrain, n)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=Z, colorscale='Viridis', showscale=False,
                             opacity=0.8, hoverinfo='skip', name='loss'))
    off = 0.03 * float(np.nanmax(Z))                 # lift lines above the surface
    fig.add_trace(go.Scatter3d(
        x=[terrain.minimum[0]], y=[terrain.minimum[1]], z=[off], mode='markers',
        marker=dict(size=7, color='gold', symbol='diamond'), name='Minimum'))
    end_symbol = {'finished': 'diamond', 'crashed': 'x'}
    for log in logs:
        c = colors[log.name]
        px, py = _plot_path(terrain, log)
        p = torch.tensor([px, py], dtype=torch.float32)
        gap = torch.clamp(terrain.fn(p) - terrain.fmin, min=0.0)
        pz = (torch.log1p(gap).numpy() + off).tolist()
        fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines',
                                   line=dict(color=c, width=6), name=log.name))
        fig.add_trace(go.Scatter3d(x=px[:1], y=py[:1], z=pz[:1], mode='markers',
                                   marker=dict(size=5, color=c, symbol='circle'),
                                   showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(
            x=px[-1:], y=py[-1:], z=pz[-1:], mode='markers',
            marker=dict(size=7, color=c, symbol=end_symbol.get(log.status, 'square')),
            showlegend=False,
            hovertext=f'{log.name}: {log.status}, step {log.steps[-1]}',
            hoverinfo='text'))
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(xaxis_title='x (weight 1)', yaxis_title='y (weight 2)',
                   zaxis_title='log(1 + loss)'),
        legend=dict(orientation='h', y=1.05))
    return fig