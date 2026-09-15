import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from ..ingestion import DataError

MAX_CELLS=10_000_000


def feature_schema(frame, columns):
    if not isinstance(columns,list) or not 1<=len(columns)<=50 or any(not isinstance(c,str) for c in columns):
        raise DataError('Select 1–50 feature columns.')
    if len(set(columns))!=len(columns) or not set(columns)<=set(frame):
        raise DataError('Select distinct existing feature columns.')
    dates=[c for c in columns if pd.api.types.is_datetime64_any_dtype(frame[c])]
    if dates:
        raise DataError('Convert datetime features to meaningful numeric/category values before modeling: '+', '.join(dates))
    return {c:'numeric' if pd.api.types.is_numeric_dtype(frame[c]) else 'categorical' for c in columns}


def prepare_features(frame, schema):
    missing=[c for c in schema if c not in frame]
    if missing:raise DataError('Prediction data is missing required features: '+', '.join(missing))
    output=pd.DataFrame(index=frame.index)
    for c,kind in schema.items():
        if kind=='numeric':
            converted=pd.to_numeric(frame[c],errors='coerce')
            if (frame[c].notna() & converted.isna()).any():
                raise DataError(f'{c} contains text incompatible with its trained numeric type.')
            output[c]=converted.astype(float).replace([np.inf,-np.inf],np.nan)
        else:
            output[c]=frame[c].map(lambda v:np.nan if pd.isna(v) else str(v)).astype(object)
    return output


def preprocessor(schema, scaling, nonnegative=False):
    if scaling not in {'standard','minmax','robust','none'}:raise DataError('Choose a listed scaling method.')
    numeric=[c for c,k in schema.items() if k=='numeric']
    categorical=[c for c,k in schema.items() if k=='categorical']
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
    if nonnegative:steps.append(('scale',MinMaxScaler(clip=True)))
    elif scaling!='none':steps.append(('scale',{'standard':StandardScaler,'minmax':MinMaxScaler,'robust':RobustScaler}[scaling]()))
    transformers=[]
    if numeric:transformers.append(('numeric',Pipeline(steps),numeric))
    if categorical:transformers.append(('category',Pipeline([
        ('impute',SimpleImputer(strategy='constant',fill_value='[missing]',keep_empty_features=True)),
        ('encode',OneHotEncoder(handle_unknown='ignore',max_categories=32,sparse_output=False))]),categorical))
    return ColumnTransformer(transformers,verbose_feature_names_out=True)


def size_guard(data, schema, degree=None):
    width=sum(1 if k=='numeric' else min(32,data[c].nunique(dropna=False)+1) for c,k in schema.items())
    if degree:width=math.comb(width+degree,degree)-1
    if width>2000 or len(data)*width>MAX_CELLS:
        raise DataError('Encoded design exceeds 2,000 features or 10 million cells. Use fewer features, categories, rows, or a lower polynomial degree.')
    return width


def make_pipeline(schema, options, model, params):
    steps=[('preprocess',preprocessor(schema,options.get('scaling','standard'),options['model']=='multinomial_nb'))]
    if options['model']=='polynomial':steps.append(('polynomial',PolynomialFeatures(degree=params['degree'],include_bias=False)))
    steps.append(('model',model))
    return Pipeline(steps)
