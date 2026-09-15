"""Report helpers shared by worker-only statistical analyses."""
import html
import json
import math
import warnings

import numpy as np
import pandas as pd
from plotly.utils import PlotlyJSONEncoder
from ..ingestion import DataError

MAX_COLUMNS = 30
PLOT_ROWS = 5000


def number(value):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return number(value)
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def report(title, kind):
    return {'title': title, 'analysis_type': kind, 'metrics': [], 'tables': [], 'figures': [],
            'warnings': [], 'notes': [], 'selection': {}}


def table(title, rows, columns=None):
    rows = list(rows)
    return {'title': title, 'columns': columns or (list(rows[0]) if rows else []), 'rows': rows}


def dataframe_table(title, frame):
    return table(title, json.loads(frame.to_json(orient='records', date_format='iso', double_precision=15)),
                 [str(c) for c in frame.columns])


def figure(title, fig):
    fig.update_layout(template='plotly_white', title=html.escape(title), height=360,
                      font={'family': 'Segoe UI, sans-serif', 'size': 12},
                      colorway=['#237a60', '#d99a35', '#7c6bb1', '#3c81a0', '#c75f69'],
                      margin={'l': 55, 'r': 20, 't': 55, 'b': 55})
    return {'title': title, 'figure': json.loads(json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder))}


def selected(frame, options, result, numeric_only=False, minimum=1):
    columns = options.get('columns')
    if not isinstance(columns, list) or not columns or any(not isinstance(c, str) for c in columns):
        raise DataError('Select columns before running an analysis.')
    columns = list(dict.fromkeys(columns))
    if not set(columns) <= set(frame.columns):
        raise DataError('Some selected columns are no longer in this dataset. Select columns again.')
    requested = columns.copy()
    excluded = []
    if numeric_only:
        excluded = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
        columns = [c for c in columns if c not in excluded]
    if len(columns) < minimum:
        suffix = ' Non-numeric columns excluded: ' + ', '.join(excluded) if excluded else ''
        raise DataError(f'Select at least {minimum} applicable column(s).' + suffix)
    if len(columns) > MAX_COLUMNS:
        raise DataError(f'Select at most {MAX_COLUMNS} columns per analysis.')
    result['selection'] = {'requested': requested, 'used': columns, 'excluded': excluded}
    if excluded:
        result['warnings'].append('Non-numeric columns excluded: ' + ', '.join(excluded))
    data = frame[columns].copy()
    for c in data.select_dtypes(include='number').columns:
        data[c] = data[c].replace([np.inf, -np.inf], np.nan)
    return data


def complete_rows(data, result):
    clean = data.dropna().copy()
    result['metrics'].extend([{'name': 'Rows used', 'value': len(clean)},
                              {'name': 'Rows excluded', 'value': len(data) - len(clean)}])
    if len(clean) < len(data):
        result['warnings'].append(f'{len(data)-len(clean):,} rows with missing/non-finite selected values were excluded.')
    return clean


def sample_plot(data, result):
    if len(data) > PLOT_ROWS:
        result['notes'].append(f'Point/distribution plots use a deterministic sample of {PLOT_ROWS:,} of {len(data):,} rows. Statistical tables use all applicable rows.')
        return data.sample(PLOT_ROWS, random_state=42).sort_index()
    return data


def diagnostic(name, callback, notes=''):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            statistic, pvalue = callback()
        return {'Test': name, 'Statistic': number(statistic), 'p-value': number(pvalue), 'Notes': notes}
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError, AssertionError):
        return {'Test': name, 'Statistic': None, 'p-value': None,
                'Notes': 'Unavailable for this sample size or design.'}
