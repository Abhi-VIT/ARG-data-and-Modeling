import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from .common import report, selected, table, figure, number
from ..ingestion import DataError

NUMERIC_METHODS = {'pearson', 'spearman', 'kendall'}
METHODS = NUMERIC_METHODS | {'auto', 'point_biserial', 'cramers_v', 'phi'}


def binary_codes(series):
    order = sorted(series.unique(), key=str)
    if len(order) != 2:
        raise ValueError('Not binary')
    return series.map({order[0]: 0, order[1]: 1}).astype(float), order


def suggest(x, y):
    nx, ny = x.nunique(), y.nunique()
    numeric_x, numeric_y = pd.api.types.is_numeric_dtype(x), pd.api.types.is_numeric_dtype(y)
    if nx == ny == 2:
        return 'phi'
    if (nx == 2 and numeric_y) or (ny == 2 and numeric_x):
        return 'point_biserial'
    if numeric_x and numeric_y:
        return 'pearson'
    if not numeric_x and not numeric_y:
        return 'cramers_v'
    return None


def correlate_pair(x, y, method):
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return None, None, 'Need at least 3 paired observations and variation in both columns.'
    if method in NUMERIC_METHODS:
        if not pd.api.types.is_numeric_dtype(x) or not pd.api.types.is_numeric_dtype(y):
            return None, None, 'This method requires two numeric columns.'
        fn = {'pearson': stats.pearsonr, 'spearman': stats.spearmanr, 'kendall': stats.kendalltau}[method]
        coefficient, pvalue = fn(x.astype(float), y.astype(float))
        return number(coefficient), number(pvalue), ''
    if method == 'point_biserial':
        if y.nunique() == 2 and (x.nunique() != 2 or (not pd.api.types.is_numeric_dtype(y) and pd.api.types.is_numeric_dtype(x))):
            x, y = y, x
        if x.nunique() != 2 or not pd.api.types.is_numeric_dtype(y):
            return None, None, 'Point-biserial requires a binary column and a numeric column.'
        codes, order = binary_codes(x)
        coefficient, pvalue = stats.pointbiserialr(codes, y.astype(float))
        return number(coefficient), number(pvalue), f'Binary coding: {order[0]} = 0; {order[1]} = 1.'
    if method == 'phi':
        if x.nunique() != 2 or y.nunique() != 2:
            return None, None, 'Phi requires two binary columns.'
        a, ax = binary_codes(x)
        b, bx = binary_codes(y)
        coefficient = stats.pearsonr(a, b).statistic
        contingency = pd.crosstab(a, b)
        _, pvalue, _, expected = stats.chi2_contingency(contingency, correction=False)
        note = f'0/1 coding: {ax}; {bx}. Chi-square p-value without continuity correction.'
        if (expected < 5).any():
            note += ' Expected counts below 5; asymptotic p-value may be unreliable.'
        return number(coefficient), number(pvalue), note
    if method == 'cramers_v':
        if max(x.nunique(), y.nunique()) > 50:
            return None, None, 'Cramér’s V is limited to 50 levels per column. Bin continuous data first.'
        contingency = pd.crosstab(x, y)
        chi, pvalue, _, expected = stats.chi2_contingency(contingency, correction=False)
        coefficient = np.sqrt(chi / (len(x) * (min(contingency.shape) - 1)))
        note = 'Uncorrected Cramér’s V (nonnegative association).'
        if (expected < 5).any():
            note += ' Expected counts below 5; asymptotic p-value may be unreliable.'
        return number(coefficient), number(pvalue), note
    return None, None, 'No supported automatic method for this pair; choose a method or recode columns.'


def correlation(frame, options, progress):
    result = report('Correlation & association', 'correlation')
    method = options.get('method', 'auto')
    if method not in METHODS:
        raise DataError('Choose a supported correlation method.')
    data = selected(frame, options, result, numeric_only=method in NUMERIC_METHODS, minimum=2)
    columns = list(data.columns)
    count = len(columns)
    coefficients = [[None] * count for _ in columns]
    pvalues = [[None] * count for _ in columns]
    observations = [[0] * count for _ in columns]
    details = []
    for i, a in enumerate(columns):
        usable = data[a].dropna()
        coefficients[i][i] = 1.0 if usable.nunique() > 1 and len(usable) >= 3 else None
        observations[i][i] = len(usable)
        for j in range(i + 1, count):
            b = columns[j]
            pair = data[[a, b]].dropna()
            recommendation = suggest(pair[a], pair[b])
            chosen = recommendation if method == 'auto' else method
            value, pvalue, note = correlate_pair(pair[a], pair[b], chosen)
            coefficients[i][j] = coefficients[j][i] = value
            pvalues[i][j] = pvalues[j][i] = pvalue
            observations[i][j] = observations[j][i] = len(pair)
            details.append({'Column A': a, 'Column B': b, 'Suggested': recommendation or 'Recode/select manually',
                            'Used': chosen or 'Unavailable', 'Coefficient': value, 'p-value': pvalue,
                            'Paired n': len(pair), 'Notes': note})
        progress(30 + int(50 * (i + 1) / count), f'Comparing column {i+1} of {count}')
    for title, matrix in [('Coefficient matrix', coefficients), ('p-value matrix', pvalues), ('Pairwise observation counts', observations)]:
        # Fixed labels avoid collisions with dataset columns called "Column".
        result['tables'].append(table(title, [{'Column': name, **{f'[{j+1}] {c}': row[j] for j, c in enumerate(columns)}}
                                             for name, row in zip(columns, matrix)]))
    result['tables'].append(table('Pairwise methods & interpretation', details))
    labels = [html.escape(c) for c in columns]
    result['figures'].append(figure('Correlation heatmap', go.Figure(go.Heatmap(
        z=coefficients, x=labels, y=labels, zmin=-1, zmax=1, colorscale='BrBG',
        hovertemplate='%{y} / %{x}<br>Association: %{z:.4f}<extra></extra>'))))
    result['metrics'] = [{'name': 'Selected columns', 'value': count}, {'name': 'Column pairs', 'value': len(details)}]
    result['notes'].append('Pairwise complete observations; missing and non-finite values are excluded per pair. p-values are unadjusted for multiple comparisons; the diagonal has no p-value.')
    if method == 'auto':
        result['notes'].append('Auto uses Phi for two binary variables, point-biserial for binary/numeric, Pearson for numeric/numeric, and Cramér’s V for two categorical variables. Coefficients from different methods have different interpretations.')
    return result
