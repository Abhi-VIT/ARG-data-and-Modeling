"""Deterministic, serializable cleaning operations; no eval or user code."""
import numpy as np
import pandas as pd
import regex
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from .ingestion import DataError, normalize_frame


def selection(frame, operation):
    scope = operation.get('selection') or {}
    columns = scope.get('columns') or list(frame.columns)
    if not isinstance(columns, list) or not set(columns) <= set(frame.columns):
        raise DataError('Select existing columns.')
    columns = list(dict.fromkeys(columns))
    rows = scope.get('rows')
    if scope.get('range') is not None:
        bounds = scope['range']
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise DataError('Row range requires a start and end.')
        start, end = map(int, bounds)
        if start < 0 or end < start or end >= len(frame):
            raise DataError('Row range is outside the dataset.')
        rows = list(range(start, end + 1))
    if rows is None:
        rows = frame.index
    elif not isinstance(rows, list) or any(type(r) is not int or r < 0 or r >= len(frame) for r in rows):
        raise DataError('Selected rows are outside this dataset revision.')
    return columns, list(dict.fromkeys(rows))


def numeric(frame, columns):
    invalid = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if invalid:
        raise DataError('Select numeric columns for this operation. Non-numeric: ' + ', '.join(invalid[:10]))


def bounded_number(value, low, high, label):
    number = float(value)
    if not np.isfinite(number) or not low <= number <= high:
        raise DataError(f'{label} must be between {low} and {high}.')
    return number


def flags(frame, operation):
    columns, rows = selection(frame, operation)
    if operation['action'] == 'detect_duplicates':
        mask = frame.duplicated(subset=columns, keep=False)
    else:
        numeric(frame, columns)
        values = frame[columns].astype(float).replace([np.inf, -np.inf], np.nan)
        method = operation.get('method', 'iqr')
        if method == 'iqr':
            k = bounded_number(operation.get('threshold', 1.5), 0.1, 10, 'IQR multiplier')
            q1, q3 = values.quantile(.25), values.quantile(.75)
            mask = ((values < q1 - k * (q3 - q1)) | (values > q3 + k * (q3 - q1))).any(axis=1)
        elif method == 'zscore':
            k = bounded_number(operation.get('threshold', 3), 0.1, 20, 'Z-score threshold')
            mask = ((values - values.mean()).abs() / values.std(ddof=0).replace(0, np.nan) > k).any(axis=1)
        elif method == 'isolation_forest':
            contamination = bounded_number(operation.get('contamination', .05), .001, .5, 'Contamination')
            if len(frame) > 100_000:
                raise DataError('Isolation forest is capped at 100,000 rows for this single-worker release.')
            if values.isna().all().any() or len(frame) < 2:
                raise DataError('Isolation forest needs at least two rows and nonempty numeric columns.')
            filled = values.fillna(values.median())
            mask = pd.Series(IsolationForest(contamination=contamination, random_state=42, n_jobs=1)
                             .fit_predict(filled) == -1, index=frame.index)
        else:
            raise DataError('Choose IQR, Z-score or isolation forest.')
    return mask & frame.index.isin(rows)


def apply_operation(frame, operation):
    if not isinstance(operation, dict):
        raise DataError('Each recipe operation must be an object.')
    frame = frame.copy()
    columns, rows = selection(frame, operation)
    action = operation.get('action')
    method = operation.get('method', '')
    column_wide = {'drop_selected_columns', 'drop_columns', 'convert', 'encode', 'scale', 'rename', 'split', 'merge'}
    if action in column_wide and len(rows) != len(frame):
        raise DataError('This operation changes whole columns. Clear the row/cell selection first.')
    if action == 'edit':
        if len(columns) != 1 or len(rows) != 1:
            raise DataError('Select exactly one cell to edit.')
        column, row = columns[0], rows[0]
        value = operation.get('value')
        if value is not None:
            if pd.api.types.is_numeric_dtype(frame[column]):
                value = pd.to_numeric(value, errors='raise')
                if pd.api.types.is_integer_dtype(frame[column]) and (not np.isfinite(value) or value != int(value)):
                    frame[column] = frame[column].astype(float)
            elif pd.api.types.is_datetime64_any_dtype(frame[column]):
                value = pd.to_datetime(value, errors='raise')
            elif isinstance(frame[column].dtype, pd.CategoricalDtype) and value not in frame[column].cat.categories:
                frame[column] = frame[column].cat.add_categories([value])
        frame.loc[row, column] = value
    elif action == 'fill':
        if method in {'mean', 'median', 'interpolate', 'knn'}:
            numeric(frame, columns)
        values = frame[columns].copy()
        if method in {'mean', 'median', 'interpolate', 'knn'}:
            # Nullable integer arrays cannot hold fractional imputations.
            values = values.astype(float)
        if method == 'knn':
            if len(frame) > 10_000 or len(columns) > 50:
                raise DataError('KNN imputation is limited to 10,000 rows and 50 selected columns. Use median for larger data.')
            if values.isna().all().any():
                raise DataError('KNN cannot impute an entirely empty column. Fill it with a constant first.')
            k = int(bounded_number(operation.get('neighbors', 5), 1, 50, 'Neighbors'))
            filled = pd.DataFrame(KNNImputer(n_neighbors=k).fit_transform(values), columns=columns, index=frame.index)
        elif method in {'mean', 'median'}:
            filled = values.fillna(getattr(values, method)())
        elif method == 'mode':
            modes = values.mode()
            if modes.empty:
                raise DataError('Entirely empty columns need a constant fill value.')
            filled = values.fillna(modes.iloc[0])
        elif method == 'constant':
            value = operation.get('value')
            if value is None:
                raise DataError('Enter a non-null fill value.')
            filled = values.copy()
            for c in columns:
                v = pd.to_numeric(value, errors='raise') if pd.api.types.is_numeric_dtype(values[c]) else str(value)
                if isinstance(filled[c].dtype, pd.CategoricalDtype) and v not in filled[c].cat.categories:
                    filled[c] = filled[c].cat.add_categories([v])
                filled[c] = filled[c].fillna(v)
        elif method in {'ffill', 'bfill'}:
            filled = getattr(values, method)()
        elif method == 'interpolate':
            filled = values.interpolate(method='linear', limit_area='inside')
        elif method == 'group_mode':
            group = operation.get('group')
            if group not in frame.columns or group in columns:
                raise DataError('Choose a grouping column outside the selected fill columns.')
            filled = values.copy()
            for c in columns:
                modes = frame.groupby(group, dropna=False, observed=True)[c].transform(
                    lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
                filled[c] = values[c].fillna(modes)
        else:
            raise DataError('Choose a supported missing-value method.')
        for c in columns:
            # Use the result dtype so mean fills may promote integer columns safely.
            result = frame[c].astype(filled[c].dtype).copy()
            result.loc[rows] = filled.loc[rows, c]
            frame[c] = result
    elif action == 'drop_missing':
        if method not in {'any', 'all'}:
            raise DataError('Choose any or all missing values.')
        mask = getattr(frame[columns].isna(), method)(axis=1) & frame.index.isin(rows)
        frame = frame.loc[~mask]
    elif action == 'drop_selected_columns':
        if not (operation.get('selection') or {}).get('columns'):
            raise DataError('Select the columns you want to drop.')
        if len(columns) >= len(frame.columns):
            raise DataError('Keep at least one column. Use Clear dataset to remove the whole dataset.')
        frame = frame.drop(columns=columns)
    elif action == 'drop_columns':
        threshold = bounded_number(operation.get('threshold', 50), 0, 100, 'Missing percent')
        frame = frame.drop(columns=[c for c in columns if frame[c].isna().mean() * 100 > threshold])
    elif action == 'drop_duplicates':
        mask = frame.duplicated(subset=columns, keep='first') & frame.index.isin(rows)
        frame = frame.loc[~mask]
    elif action == 'remove_outliers':
        frame = frame.loc[~flags(frame, operation)]
    elif action == 'convert':
        for c in columns:
            if method == 'numeric':
                frame[c] = pd.to_numeric(frame[c], errors='raise')
            elif method == 'datetime':
                frame[c] = pd.to_datetime(frame[c], format=operation.get('format') or 'mixed', errors='raise', utc=True)
            elif method in {'string', 'category'}:
                frame[c] = frame[c].astype(method)
            else:
                raise DataError('Choose string, numeric, datetime or category.')
    elif action == 'text':
        pattern = operation.get('pattern', '')
        if method == 'regex' and (not pattern or len(pattern) > 200):
            raise DataError('Regex patterns must contain 1–200 characters.')
        compiled = regex.compile(pattern) if method == 'regex' else None
        for c in columns:
            if pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_datetime64_any_dtype(frame[c]):
                raise DataError('Convert selected columns to string before text cleaning.')
            s = frame[c].astype('string')
            if method == 'trim':
                result = s.str.strip()
            elif method in {'lower', 'upper', 'title'}:
                result = getattr(s.str, method)()
            elif method == 'special':
                result = s.map(lambda v: regex.sub(r'[^\p{L}\p{N}\s]', '', v, timeout=.05) if pd.notna(v) else v)
            elif method == 'regex':
                replacement = str(operation.get('replacement', ''))
                if len(replacement) > 1000:
                    raise DataError('Replacement is too long.')
                result = s.map(lambda v: compiled.sub(replacement, v, timeout=.05) if pd.notna(v) else v)
            else:
                raise DataError('Choose trim, lower, upper, title, regex or special-character removal.')
            s.loc[rows] = result.loc[rows]
            frame[c] = s
    elif action == 'encode':
        if method == 'onehot':
            if sum(frame[c].nunique() + 1 for c in columns) + len(frame.columns) - len(columns) > 500:
                raise DataError('One-hot encoding would exceed 500 columns.')
            frame = pd.get_dummies(frame, columns=columns, dummy_na=True, dtype=int)
        elif method in {'label', 'ordinal'}:
            for c in columns:
                order = operation.get('order') if method == 'ordinal' else sorted(frame[c].dropna().unique(), key=str)
                if not isinstance(order, list) or len(set(order)) != len(order):
                    raise DataError('Ordinal order must be a JSON list of distinct values.')
                if not set(frame[c].dropna()) <= set(order):
                    raise DataError('Ordinal order must include every observed non-null value.')
                frame[c] = frame[c].map({value: i for i, value in enumerate(order)}).astype('Int64')
        elif method == 'target':
            target = operation.get('target')
            if target not in frame.columns or target in columns:
                raise DataError('Choose a numeric target outside the encoded columns.')
            numeric(frame, [target])
            for c in columns:
                means = frame.groupby(c, observed=True, dropna=False)[target].mean()
                frame[c] = frame[c].map(means).astype(float)
        else:
            raise DataError('Choose one-hot, label, ordinal or target encoding.')
    elif action == 'scale':
        numeric(frame, columns)
        if not len(frame):
            raise DataError('Scaling needs at least one row.')
        if method == 'log':
            if (frame[columns] <= -1).any().any():
                raise DataError('log1p requires every selected value to be greater than -1.')
            frame[columns] = np.log1p(frame[columns])
        else:
            scalers = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}
            if method not in scalers:
                raise DataError('Choose standard, minmax, robust or log.')
            frame[columns] = scalers[method]().fit_transform(frame[columns])
    elif action == 'rename':
        if len(columns) != 1:
            raise DataError('Select one column to rename.')
        name = str(operation.get('name', '')).strip()
        if not name or name in frame.columns:
            raise DataError('Enter a new, unique column name.')
        frame = frame.rename(columns={columns[0]: name})
    elif action == 'split':
        delimiter = operation.get('delimiter', ',')
        if len(columns) != 1 or not delimiter:
            raise DataError('Select one column and enter a literal delimiter.')
        parts = frame[columns[0]].astype('string').str.split(delimiter, n=19, expand=True, regex=False)
        parts.columns = [f'{columns[0]}_{i+1}' for i in range(len(parts.columns))]
        if set(parts.columns) & set(frame.columns):
            raise DataError('Split column names already exist.')
        frame = pd.concat([frame, parts], axis=1)
    elif action == 'merge':
        name = str(operation.get('name', '')).strip()
        if len(columns) < 2 or not name or name in frame.columns:
            raise DataError('Select two or more columns and a new output name.')
        delimiter = str(operation.get('delimiter', ' '))
        frame[name] = frame[columns].astype('string').fillna('').agg(delimiter.join, axis=1)
    else:
        raise DataError('Unsupported cleaning operation.')
    return normalize_frame(frame)
