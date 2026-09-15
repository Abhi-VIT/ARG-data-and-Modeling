import json
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from django.conf import settings
from .ingestion import ROW_ID, normalize_frame


def private_path(workspace_id, suffix):
    directory = settings.MEDIA_ROOT / str(workspace_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f'{uuid.uuid4().hex}{suffix}'


def profile(frame):
    columns = []
    for name in frame.columns:
        series = frame[name]
        info = {'name': name, 'dtype': str(series.dtype), 'missing': int(series.isna().sum()),
                'missing_pct': round(float(series.isna().mean() * 100), 2) if len(frame) else 0,
                'unique': int(series.nunique()), 'histogram': None}
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna().to_numpy(dtype=float)
            clean = clean[np.isfinite(clean)]
            if len(clean):
                counts, edges = np.histogram(clean, bins=16)
                info['histogram'] = {'x': ((edges[:-1] + edges[1:]) / 2).tolist(), 'y': counts.tolist()}
        columns.append(info)
    boundaries = np.linspace(0, len(frame), min(40, max(1, len(frame))) + 1, dtype=int)
    heatmap = [frame.iloc[start:end].isna().mean().tolist() if end > start else [0] * len(columns)
               for start, end in zip(boundaries[:-1], boundaries[1:])]
    return {'rows': len(frame), 'columns': columns, 'column_count': len(columns),
            'memory_bytes': int(frame.memory_usage(deep=True).sum()),
            'missing': sum(c['missing'] for c in columns), 'duplicates': int(frame.duplicated().sum()),
            'dtypes': {str(k): int(v) for k, v in frame.dtypes.astype(str).value_counts().items()},
            'heatmap': heatmap}


def write_frame(workspace_id, frame):
    frame = normalize_frame(frame)
    path = private_path(workspace_id, '.parquet')
    frame.assign(**{ROW_ID: np.arange(len(frame), dtype=np.int64)}).to_parquet(path, index=False)
    return str(path), profile(frame)


def read_frame(path):
    return pd.read_parquet(path).drop(columns=ROW_ID).reset_index(drop=True)


def safe_spreadsheet(frame):
    """Neutralize formulas in spreadsheet-oriented text exports."""
    frame = frame.copy()
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else v)
    frame.columns = ["'" + c if c.lstrip().startswith(('=', '+', '-', '@')) else c for c in frame.columns]
    return frame


def export_frame(frame, path, format):
    if format in {'csv', 'tsv'}:
        safe_spreadsheet(frame).to_csv(path, sep='\t' if format == 'tsv' else ',', index=False)
    elif format == 'json':
        frame.to_json(path, orient='records', date_format='iso', indent=2)
    elif format == 'parquet':
        frame.to_parquet(path, index=False)
    elif format == 'xlsx':
        frame = safe_spreadsheet(frame)
        # Excel has no timezone-aware cell type. Export UTC wall time explicitly.
        for column in frame.select_dtypes(include=['datetimetz']).columns:
            frame[column] = frame[column].dt.tz_convert('UTC').dt.tz_localize(None)
        frame.to_excel(path, index=False, engine='openpyxl')
    else:
        raise ValueError('Choose CSV, TSV, JSON, Parquet or XLSX.')
