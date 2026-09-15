import json
import warnings
from importlib.metadata import version
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler, KFold, StratifiedKFold, cross_val_score
from threadpoolctl import threadpool_limits
from ..ingestion import DataError
from ..analytics.common import report, table, figure, number, clean_json
from .catalog import CATALOG, validate_params, estimator
from .preprocessing import feature_schema, prepare_features, make_pipeline, size_guard
from .evaluation import supervised_report, projection_plot


def bounded(value,low,high,label,integer=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not low<=value<=high or (integer and value!=int(value)):
        raise DataError(f'{label} must be between {low} and {high}.')
    return int(value) if integer else float(value)


def tune(pipeline,key,params,options,x,y,task,seed,result,progress,schema):
    search=options.get('search',{'method':'none'})
    if not isinstance(search,dict) or search.get('method','none') not in {'none','grid','random'}:
        raise DataError('Choose no search, grid search or random search.')
    if search.get('method','none')=='none':return pipeline,params
    if len(x)>20_000:raise DataError('Hyperparameter search is capped at 20,000 training rows.')
    ranges=search.get('ranges')
    if not isinstance(ranges,dict) or not ranges:raise DataError('Select parameters and enter candidate values for search.')
    combinations=1
    for name,values in ranges.items():
        if not isinstance(values,list) or not 1<=len(values)<=10:raise DataError('Each search parameter needs 1–10 candidate values.')
        for value in values:validate_params(key,{name:value})
        combinations*=len(values)
    folds=bounded(search.get('folds',3),2,5,'CV folds',True)
    if search['method']=='grid':
        if combinations>20:raise DataError('Grid search supports at most 20 combinations. Narrow the ranges or use random search.')
        candidates=list(ParameterGrid(ranges))
    else:
        count=bounded(search.get('iterations',10),1,20,'Random-search iterations',True)
        candidates=list(ParameterSampler(ranges,n_iter=min(count,combinations),random_state=seed))
    if task=='classification':
        if pd.Series(y).value_counts().min()<folds:raise DataError('Every training class needs at least as many rows as CV folds.')
        cv=StratifiedKFold(folds,shuffle=True,random_state=seed);scoring='accuracy'
    else:
        cv=KFold(folds,shuffle=True,random_state=seed);scoring='neg_root_mean_squared_error'
    rows=[];best=None;best_score=-np.inf;best_params=None
    for i,candidate in enumerate(candidates):
        selected_params={**params,**candidate}
        if key=='polynomial':size_guard(x,schema,selected_params['degree'])
        translated={('polynomial__degree' if k=='degree' else 'model__'+k):v for k,v in candidate.items()}
        trial=clone(pipeline).set_params(**translated)
        progress(30+int(40*i/len(candidates)),f'Cross-validating candidate {i+1}/{len(candidates)} ({folds} folds)')
        try:
            scores=cross_val_score(trial,x,y,cv=cv,scoring=scoring,n_jobs=1,error_score='raise')
            score=float(np.mean(scores))
            if not np.isfinite(score):raise ValueError('Non-finite validation score.')
            rows.append({'Candidate':i+1,'Parameters':json.dumps(candidate),'Mean CV score':score,'Std. dev.':float(np.std(scores)),'Status':'Complete'})
            if score>best_score:best,best_score,best_params=trial,score,selected_params
        except ValueError as exc:
            rows.append({'Candidate':i+1,'Parameters':json.dumps(candidate),'Mean CV score':None,'Std. dev.':None,'Status':str(exc)[:180]})
    if best is None:raise DataError('No search candidate could be fitted. Check sample size, class counts and parameter ranges.')
    result['tables'].append(table('Cross-validation search results',rows))
    result['metrics'].append({'name':'Best CV score','value':best_score})
    result['notes'].append(f'Search maximizes {scoring} on training-only {folds}-fold cross-validation. Imputation, encoding and scaling are refitted in every fold. Negative RMSE is better closer to zero. The held-out test set is used only after model selection.')
    return best,best_params


def train(frame, options, progress=lambda percent,message:None):
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=2):
        warnings.simplefilter('always')
        try:result,bundle,output=_train(frame,options,progress)
        except (ValueError,TypeError,OverflowError,np.linalg.LinAlgError) as exc:
            raise DataError('Model could not fit these selections: '+str(exc)[:350]) from exc
    result['warnings'].extend(list(dict.fromkeys(str(w.message)[:300] for w in captured))[:10])
    result=clean_json(result)
    json.dumps(result,allow_nan=False)
    return result,bundle,output


def _train(frame,options,progress):
    if not isinstance(options,dict) or options.get('model') not in CATALOG:raise DataError('Choose a supported model.')
    key=options['model'];spec=CATALOG[key];task=spec['task']
    params=validate_params(key,options.get('params',{}))
    schema=feature_schema(frame,options.get('features'))
    seed=bounded(options.get('seed',42),0,2**31-1,'Random seed',True)
    cap={'hierarchical':2500,'dbscan':5000,'tsne':3000,'umap':10000,'svm':10000,'svr':10000}.get(key,50000)
    if not 8<=len(frame)<=cap:raise DataError(f'This model supports 8–{cap:,} rows. Reduce the dataset explicitly before fitting.')
    result=report(spec['name'],'ml');result['task_type']=task;result['model_key']=key
    result['selection']={'requested':list(schema),'used':list(schema),'excluded':[]}
    result['parameters']=options
    data=prepare_features(frame,schema)
    size_guard(data,schema,params.get('degree'))
    model=estimator(key,params,seed)
    pipeline=make_pipeline(schema,options,model,params)
    classes=None;target=None;output=None
    progress(25,'Preparing selected features and training pipeline')
    if task in {'regression','classification'}:
        target=options.get('target')
        if target not in frame or target in schema:raise DataError('Choose a target separate from the selected features.')
        y=frame[target]
        if task=='regression':
            if not pd.api.types.is_numeric_dtype(y):raise DataError('Regression requires a numeric target.')
            y=y.astype(float).replace([np.inf,-np.inf],np.nan)
        valid=y.notna();data=data.loc[valid];y=y.loc[valid]
        if len(data)<12 or y.nunique()<2:raise DataError('Supervised training requires 12 labeled rows and a varying target.')
        if task=='classification':
            y=y.astype(str);classes=sorted(y.unique())
            if len(classes)>30:raise DataError('Classification supports at most 30 classes. For continuous outcomes choose regression.')
            mapping={c:i for i,c in enumerate(classes)};y=y.map(mapping).astype(int)
        fraction=bounded(options.get('test_size',.2),.1,.5,'Test fraction')
        stratify=options.get('stratify',True)
        if not isinstance(stratify,bool):raise DataError('Stratification must be enabled or disabled.')
        xtrain,xtest,ytrain,ytest=train_test_split(data,y,test_size=fraction,random_state=seed,stratify=y if classes and stratify else None)
        if classes and set(ytrain)!=set(range(len(classes))):raise DataError('Training split omitted a class. Enable stratification or add labeled rows.')
        if len(xtest)<2:raise DataError('Increase the test fraction to hold out at least two rows.')
        result['metrics']=[{'name':'Training rows','value':len(xtrain)},{'name':'Test rows','value':len(xtest)},{'name':'Unlabeled rows excluded','value':int((~valid).sum())}]
        if (~valid).any():result['warnings'].append(f'{int((~valid).sum())} rows with missing/non-finite targets were excluded before splitting.')
        pipeline,params=tune(pipeline,key,params,options,xtrain,ytrain,task,seed,result,progress,schema)
        progress(75,'Fitting selected model on training rows')
        pipeline.fit(xtrain,ytrain)
        progress(88,'Evaluating held-out predictions and feature importance')
        supervised_report(result,pipeline,xtest,ytest,classes,seed,key)
    else:
        if options.get('search',{}).get('method','none')!='none':raise DataError('Cross-validation search is available for supervised models. Use the K-Means helper for cluster-count exploration.')
        values=pipeline.named_steps['preprocess'].fit_transform(data)
        if not np.isfinite(values).all():raise DataError('Preprocessing produced non-finite values. Scale or simplify features.')
        if np.max(np.std(values,axis=0))<1e-12:raise DataError('Select features with variation before clustering or reduction.')
        result['metrics']=[{'name':'Rows fitted','value':len(data)},{'name':'Encoded features','value':values.shape[1]}]
        if task=='clustering':
            if params.get('n_clusters',2)>=len(data):raise DataError('Choose fewer clusters than observations.')
            labels=model.fit_predict(values)
            unique=np.unique(labels);nonnoise=labels!=-1;clusters=np.unique(labels[nonnoise])
            score=None
            if 1<len(clusters)<int(nonnoise.sum()):
                try:score=silhouette_score(values[nonnoise],labels[nonnoise],sample_size=min(1000,int(nonnoise.sum())),random_state=seed)
                except ValueError:pass
            result['metrics'].extend([{'name':'Clusters','value':len(clusters)},{'name':'Noise rows','value':int((~nonnoise).sum())},{'name':'Silhouette (non-noise)','value':number(score)}])
            result['tables'].append(table('Cluster sizes',[{'Cluster':int(k),'Rows':int((labels==k).sum())} for k in unique]))
            coordinates=PCA(n_components=2).fit_transform(values) if values.shape[1]>=2 else np.column_stack([values[:,0],np.zeros(len(values))])
            projection_plot(result,coordinates,labels,'Clusters projected onto two principal components')
            output=pd.DataFrame({'row_number':frame.index+1,'cluster':labels})
            if key=='kmeans':
                rows=[]
                for k in range(1,min(10,len(data)-1)+1):
                    progress(45+4*k,f'K-Means helper: comparing k={k}')
                    fitted=KMeans(n_clusters=k,n_init=5,random_state=seed).fit(values)
                    silhouette=None
                    if 1<len(np.unique(fitted.labels_))<len(data):
                        try:silhouette=silhouette_score(values,fitted.labels_,sample_size=min(1000,len(data)),random_state=seed)
                        except ValueError:pass
                    rows.append({'k':k,'Inertia':float(fitted.inertia_),'Silhouette':number(silhouette)})
                result['tables'].append(table('K-Means cluster-count helper',rows))
                for label in ['Inertia','Silhouette']:
                    result['figures'].append(figure('Elbow curve' if label=='Inertia' else 'Silhouette by k',go.Figure(go.Scatter(x=[r['k'] for r in rows],y=[r[label] for r in rows],mode='lines+markers'))))
                result['notes'].append('The helper evaluates k=1–10 where possible and does not change your selected k. Inertia usually decreases with k; silhouette is sampled at up to 1,000 rows.')
            elif key=='hierarchical':
                counts=np.zeros(model.children_.shape[0]);n=len(values)
                for i,children in enumerate(model.children_):counts[i]=sum(1 if c<n else counts[c-n] for c in children)
                linkage=np.column_stack([model.children_,model.distances_,counts]).astype(float)
                tree=dendrogram(linkage,no_plot=True,truncate_mode='lastp',p=30)
                fig=go.Figure()
                for xs,ys in zip(tree['icoord'],tree['dcoord']):fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',line={'color':'#237a60'},showlegend=False))
                fig.update_xaxes(tickvals=list(range(5,10*len(tree['ivl']),10)),ticktext=tree['ivl']);fig.update_yaxes(title='Linkage distance')
                result['figures'].append(figure('Dendrogram (last 30 merged groups)',fig))
            result['notes'].append('Clustering fits all selected rows. Silhouette excludes DBSCAN noise and uses at most 1,000 observations; a dash means fewer than two usable clusters or an unavailable sample. PCA is only a display projection, not the clustering input.')
        else:
            components=params.get('n_components',2)
            if key=='pca' and components>min(values.shape):raise DataError('PCA components cannot exceed the smaller of rows and encoded features.')
            if key=='tsne' and params['perplexity']>=len(values):raise DataError('t-SNE perplexity must be smaller than the number of rows.')
            if key=='umap' and params['n_neighbors']>=len(values):raise DataError('UMAP neighbors must be smaller than the number of rows.')
            progress(45,'Fitting dimensionality reduction')
            embedding=model.fit_transform(values)
            projection_plot(result,embedding,title=spec['name']+' embedding')
            output=pd.DataFrame(embedding,columns=[f'component_{i+1}' for i in range(embedding.shape[1])]);output.insert(0,'row_number',frame.index+1)
            if key=='pca':
                ratios=model.explained_variance_ratio_
                result['tables'].append(table('Explained variance',[{'Component':i+1,'Variance ratio':float(r),'Cumulative':float(np.sum(ratios[:i+1]))} for i,r in enumerate(ratios)]))
                result['figures'].append(figure('PCA scree plot',go.Figure(go.Bar(x=list(range(1,len(ratios)+1)),y=ratios.tolist()))))
                result['metrics'].append({'name':'Explained variance retained','value':float(sum(ratios))})
            result['notes'].append('Dimensionality reduction fits all selected rows and reports an exploratory embedding, without held-out predictive metrics. Distances in t-SNE/UMAP plots do not have the same interpretation as distances in the original features.')
    can_predict=key not in {'hierarchical','dbscan','tsne'}
    result['can_predict']=can_predict
    result['best_params']=params
    result['tables'].append(table('Fitted hyperparameters',[{'Parameter':k,'Value':str(v)} for k,v in params.items()]))
    result['notes'].append('Missing numeric features use training medians; categorical features use a missing token and one-hot encoding (at most 32 outputs per input column). Unseen categories become all zeros. Select features carefully: identifiers and target-derived columns can cause leakage.')
    if key=='multinomial_nb':result['notes'].append('Multinomial NB forces train-fitted min–max scaling with clipping to keep features nonnegative; use it when a multinomial feature model is appropriate.')
    if not can_predict:result['notes'].append('This estimator has no out-of-sample predict/transform method. Its fitted artifact and full row assignments/embedding can be downloaded; new-file scoring is unavailable.')
    bundle={'format_version':1,'pipeline':pipeline,'schema':schema,'target':target,'classes':classes,'task':task,'model_key':key,
            'can_predict':can_predict,'versions':{p:version(p) for p in ['scikit-learn','numpy','pandas','joblib','xgboost','lightgbm','umap-learn']}}
    result['versions']=bundle['versions']
    return result,bundle,output
