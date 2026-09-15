import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .preprocessing import prepare_features, size_guard, MAX_CELLS
from .training import bounded
from ..ingestion import DataError


def predict(bundle, frame, threshold=.5):
    if bundle.get('format_version')!=1 or not bundle.get('can_predict'):raise DataError('This model does not support predictions on new data.')
    if not 1<=len(frame)<=50_000:raise DataError('Prediction uploads must contain 1–50,000 rows.')
    data=prepare_features(frame,bundle['schema'])
    pipeline=bundle['pipeline']
    polynomial=pipeline.named_steps.get('polynomial')
    size_guard(data,bundle['schema'],polynomial.degree if polynomial is not None else None)
    encoded_width=len(pipeline[:-1].get_feature_names_out())
    if len(data)*encoded_width>MAX_CELLS:
        raise DataError('Prediction exceeds 10 million encoded cells with this trained model. Score a smaller file.')
    with threadpool_limits(limits=2):
        if bundle['task']=='reduction':
            predicted=pipeline.transform(data)
            output=pd.DataFrame(predicted,columns=[f'component_{i+1}' for i in range(predicted.shape[1])])
        elif bundle['classes'] is not None:
            classes=bundle['classes'];probs=pipeline.predict_proba(data)
            if len(classes)==2:
                cutoff=bounded(threshold,0,1,'Classification threshold')
                indices=(probs[:,1]>=cutoff).astype(int)
            else:indices=pipeline.predict(data).astype(int)
            output=pd.DataFrame({'prediction':[classes[i] for i in indices]})
            for i,c in enumerate(classes):output[f'probability [{i+1}] {c}']=probs[:,i]
        else:output=pd.DataFrame({'prediction':pipeline.predict(data)})
    output.insert(0,'row_number',np.arange(len(frame))+1)
    return output
