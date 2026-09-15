import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .common import report, selected, sample_plot, figure, table, number
from ..ingestion import DataError


def descriptive(frame, options, progress):
    result = report('Descriptive statistics', 'descriptive')
    data = selected(frame, options, result)
    rows = []
    for column in data:
        values = data[column].dropna()
        row = {'Column': column, 'Type': str(data[column].dtype), 'Count': len(values),
               'Missing / non-finite': len(data) - len(values), 'Unique': values.nunique()}
        if pd.api.types.is_numeric_dtype(values) and len(values):
            values = values.astype(float)
            row.update({'Mean': number(values.mean()), 'Std. dev.': number(values.std()),
                        'Min': number(values.min()), '25%': number(values.quantile(.25)),
                        'Median': number(values.median()), '75%': number(values.quantile(.75)),
                        'Max': number(values.max()), 'Skewness': number(values.skew()),
                        'Excess kurtosis': number(values.kurt())})
        else:
            mode = values.mode()
            row['Mode'] = str(mode.iloc[0]) if len(mode) else None
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    result['tables'].append(table('Column summaries', [{key: row.get(key) for key in fields} for row in rows], fields))
    result['metrics'] = [{'name': 'Rows', 'value': len(data)}, {'name': 'Selected columns', 'value': len(data.columns)}]
    result['notes'].append('Standard deviation uses n−1 degrees of freedom. Statistics exclude missing/non-finite values; categorical modes show the first value when tied.')
    return result


def eda(frame, options, progress):
    result = report('Exploratory plots', 'eda')
    kind = options.get('plot', 'histogram')
    if kind not in {'histogram', 'box', 'violin', 'scatter', 'pair', 'heatmap', 'bar', 'line'}:
        raise DataError('Choose a supported plot type.')
    if kind == 'heatmap':
        from .correlation import correlation
        result = correlation(frame, dict(options, method='pearson'), progress)
        result['analysis_type'] = 'eda'
        return result
    numeric_only = kind in {'histogram', 'box', 'violin', 'scatter', 'pair'}
    data = selected(frame, options, result, numeric_only=numeric_only, minimum=2 if kind in {'scatter', 'pair', 'line'} else 1)
    if kind == 'pair' and len(data.columns) > 6:
        raise DataError('Pair plots support at most 6 numeric columns.')
    if kind in {'histogram', 'box', 'violin'} and len(data.columns) > 12:
        raise DataError('Choose at most 12 columns for distribution plots.')
    sampled = data if kind == 'bar' else sample_plot(data, result)
    if kind in {'histogram', 'box', 'violin'}:
        bins = int(options.get('bins', 30))
        if not 5 <= bins <= 100:
            raise DataError('Choose 5–100 histogram bins.')
        for column in data:
            values = sampled[column].dropna().astype(float).tolist()
            if not values:
                result['warnings'].append(f'{column}: no finite values to plot.')
                continue
            if kind == 'histogram':
                trace = go.Histogram(x=values, nbinsx=bins, name=html.escape(column))
            elif kind == 'box':
                trace = go.Box(y=values, name=html.escape(column), boxpoints='outliers')
            else:
                trace = go.Violin(y=values, name=html.escape(column), box_visible=True, meanline_visible=True)
            result['figures'].append(figure(f'{column} · {kind}', go.Figure(trace)))
    elif kind in {'scatter', 'line'}:
        x, y = options.get('x'), options.get('y')
        if x not in data or y not in data or x == y:
            raise DataError('Choose different X and Y columns from your selection.')
        if not pd.api.types.is_numeric_dtype(data[y]):
            raise DataError('The Y axis must be numeric.')
        points = sampled[[x, y]].dropna().copy()
        if not len(points):
            raise DataError('No complete points are available for the selected axes.')
        if kind == 'line':
            points = points.sort_values(x)
        xvalues = points[x].tolist()
        if not pd.api.types.is_numeric_dtype(points[x]) and not pd.api.types.is_datetime64_any_dtype(points[x]):
            xvalues = [html.escape(str(v)) for v in xvalues]
        fig = go.Figure(go.Scatter(x=xvalues, y=points[y].tolist(), mode='lines+markers' if kind == 'line' else 'markers',
                                  marker={'size': 5, 'opacity': .7}))
        fig.update_xaxes(title=html.escape(x)); fig.update_yaxes(title=html.escape(y))
        result['figures'].append(figure(f'{y} versus {x}', fig))
        if kind == 'line':
            result['notes'].append('Line points are sorted by X; equal X values are retained.')
    elif kind == 'pair':
        points = sampled.dropna()
        if not len(points):
            raise DataError('No complete rows are available for the pair plot.')
        fig = go.Figure(go.Splom(dimensions=[{'label': html.escape(c), 'values': points[c].tolist()} for c in points],
                                 marker={'size': 3, 'opacity': .5}, diagonal_visible=False))
        result['figures'].append(figure('Pair plot', fig))
        result['figures'][-1]['figure']['layout']['height'] = 600
    elif kind == 'bar':
        x = options.get('x')
        if x not in data:
            raise DataError('Choose a selected column for category counts.')
        counts = data[x].dropna().astype(str).value_counts().head(30)
        if counts.empty:
            raise DataError('No categories are available to plot.')
        result['figures'].append(figure(f'{x} · category counts', go.Figure(go.Bar(
            x=[html.escape(c) for c in counts.index], y=counts.tolist()))))
        result['tables'].append(table('Category counts (top 30)', [{'Category': c, 'Count': int(n)} for c, n in counts.items()]))
        result['notes'].append('Bar counts use all nonmissing rows and show the 30 most frequent categories.')
    if not result['figures']:
        raise DataError('No usable observations for this plot.')
    return result
