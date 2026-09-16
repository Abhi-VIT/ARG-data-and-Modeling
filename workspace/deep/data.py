import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ..ingestion import DataError


def fit_scaler(values):
    medians=np.nanmedian(values,axis=0)
    if not np.isfinite(medians).all():raise DataError('Every selected feature needs a finite value in the training split.')
    filled=np.where(np.isnan(values),medians,values)
    mean=filled.mean(axis=0);scale=filled.std(axis=0);scale[scale<1e-8]=1
    return {'median':medians.tolist(),'mean':mean.tolist(),'scale':scale.tolist()}


def transform(values,scaler):
    values=np.asarray(values,dtype=np.float64)
    filled=np.where(np.isfinite(values),values,np.asarray(scaler['median']))
    output=(filled-np.asarray(scaler['mean']))/np.asarray(scaler['scale'])
    if not np.isfinite(output).all() or np.max(np.abs(output))>1e6:raise DataError('Features overflowed after scaling. Check extreme values.')
    return output.astype(np.float32)


def split_indices(n,c,y=None,chronological=False):
    ids=np.arange(n)
    if chronological:
        train_end=int(n*(1-c['validation_size']-c['test_size']));val_end=int(n*(1-c['test_size']))
        return ids[:train_end],ids[train_end:val_end],ids[val_end:]
    try:
        trainval,test=train_test_split(ids,test_size=c['test_size'],random_state=c['seed'],stratify=y)
        train,val=train_test_split(trainval,test_size=c['validation_size']/(1-c['test_size']),random_state=c['seed'],stratify=y[trainval] if y is not None else None)
        return train,val,test
    except ValueError as exc:raise DataError('Cannot create three splits. Add rows per class or adjust the split fractions. '+str(exc)) from exc


def tabular_data(frame,c):
    features=c.get('features');target=c.get('target');kind=c['architecture'];task=c['task']
    if not isinstance(features,list) or not 1<=len(features)<=50 or any(not isinstance(f,str) for f in features) or len(set(features))!=len(features) or not set(features)<=set(frame):
        raise DataError('Select 1–50 distinct feature columns.')
    invalid=[f for f in features if not pd.api.types.is_numeric_dtype(frame[f])]
    if invalid:raise DataError('Neural tabular inputs require numeric features. Encode categories first: '+', '.join(invalid))
    if not 20<=len(frame)<=20_000:raise DataError('Neural tabular training supports 20–20,000 rows.')
    sequence=kind in {'rnn','lstm','gru'}
    work=frame.copy();notes=[]
    if sequence:
        order=c.get('order_column')
        if order:
            if order not in work or work[order].isna().any() or work[order].duplicated().any():raise DataError('Sequence order must be an existing column with unique, nonmissing values.')
            work=work.sort_values(order,kind='stable')
        notes.append('Sequence rows are ordered by '+(order or 'the current dataset order')+'. This is one continuous series; windows overlap within each split, but no window crosses split boundaries. Each window predicts the next row.')
    if task!='reconstruction':
        if target not in frame or target in features:raise DataError('Choose a target separate from the input features.')
        y=work[target]
        if task=='regression':
            if not pd.api.types.is_numeric_dtype(y):raise DataError('Regression targets must be numeric.')
            y=y.astype(float).replace([np.inf,-np.inf],np.nan)
        if sequence and y.isna().any():raise DataError('Sequence targets cannot be missing; dropping them would change temporal spacing.')
        valid=y.notna();work=work.loc[valid];y=y.loc[valid]
        if not valid.all():notes.append(f'{int((~valid).sum())} rows with missing targets were excluded before splitting.')
        if task=='classification':
            y=y.astype(str);classes=sorted(y.unique())
            if not 2<=len(classes)<=20:raise DataError('Classification requires 2–20 classes.')
            y=y.map({label:i for i,label in enumerate(classes)}).to_numpy(dtype=np.int64)
        else:
            classes=None;y=y.to_numpy(dtype=np.float32)
            if not np.isfinite(y).all() or np.std(y)<1e-8:raise DataError('Target needs finite variation representable in float32.')
    else:classes=None;y=None
    values=work[features].astype(float).replace([np.inf,-np.inf],np.nan).to_numpy()
    splits=split_indices(len(work),c,y if classes else None,sequence)
    minimum=c['sequence_length']+2 if sequence else 2
    if any(len(ids)<minimum for ids in splits):raise DataError(f'Every split needs at least {minimum} rows. Reduce window length or adjust the split fractions.')
    if sequence and sum(len(ids)-c['sequence_length'] for ids in splits)*c['sequence_length']*len(features)>10_000_000:
        raise DataError('Sequence design exceeds 10 million input cells. Reduce windows, rows, or features.')
    scaler=fit_scaler(values[splits[0]])
    scaled=transform(values,scaler)
    target_scaler=None
    if task=='regression':
        target_scaler={'mean':float(y[splits[0]].mean()),'scale':float(y[splits[0]].std())}
        if target_scaler['scale']<1e-8:raise DataError('Training targets need variation.')
        y=(y-target_scaler['mean'])/target_scaler['scale']
    if classes and set(y[splits[0]])!=set(range(len(classes))):raise DataError('The training split must contain every class.')
    datasets=[];rowids=[]
    for ids in splits:
        if sequence:
            length=c['sequence_length']
            x=np.stack([scaled[ids[i-length:i]] for i in range(length,len(ids))])
            target_ids=ids[length:];labels=y[target_ids]
        else:x=scaled[ids];target_ids=ids;labels=scaled[ids] if task=='reconstruction' else y[ids]
        datasets.append((x,np.asarray(labels)));rowids.append((work.index.to_numpy()[target_ids]+1).tolist())
    if sum(x.size for x,_ in datasets)>10_000_000:raise DataError('Sequence design exceeds 10 million input cells. Reduce windows, rows, or features.')
    if kind=='autoencoder' and c['latent_dim']>=len(features):raise DataError('Choose a latent dimension smaller than the number of input features.')
    return datasets,{'features':features,'scaler':scaler,'target_scaler':target_scaler,'classes':classes,'row_ids':rowids,
                     'input_width':len(features),'output_width':len(classes) if classes else (len(features) if task=='reconstruction' else 1),'notes':notes}
