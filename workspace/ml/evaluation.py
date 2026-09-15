import html
import numpy as np
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.inspection import permutation_importance
from ..analytics.common import figure, table, number


def threshold_counts(y, probabilities, threshold):
    predicted=probabilities>=threshold
    tn,fp,fn,tp=metrics.confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return {'threshold':float(threshold),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'accuracy':float((tn+tp)/len(y)),'precision':float(tp/(tp+fp)) if tp+fp else 0.,
            'recall':float(tp/(tp+fn)) if tp+fn else 0.,'f1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.}


def supervised_report(result, pipeline, x, y, classes, seed, key):
    predicted=pipeline.predict(x)
    sampled=np.linspace(0,len(x)-1,min(2000,len(x)),dtype=int)
    if classes is None:
        values={'RMSE':metrics.root_mean_squared_error(y,predicted),'MAE':metrics.mean_absolute_error(y,predicted),'R²':metrics.r2_score(y,predicted)}
        fig=go.Figure(go.Scatter(x=np.asarray(y)[sampled].tolist(),y=predicted[sampled].tolist(),mode='markers',marker={'size':5},name='Held-out observations'))
        fig.update_xaxes(title='Actual');fig.update_yaxes(title='Predicted')
        result['figures'].append(figure('Held-out predictions',fig))
    else:
        labels=list(range(len(classes)))
        average='binary' if len(classes)==2 else 'weighted'
        values={'Accuracy':metrics.accuracy_score(y,predicted),'Precision':metrics.precision_score(y,predicted,average=average,zero_division=0),
                'Recall':metrics.recall_score(y,predicted,average=average,zero_division=0),'F1':metrics.f1_score(y,predicted,average=average,zero_division=0)}
        probs=pipeline.predict_proba(x)
        try:
            values['ROC-AUC']=metrics.roc_auc_score(y,probs[:,1]) if len(classes)==2 else metrics.roc_auc_score(y,probs,multi_class='ovr',average='weighted',labels=labels)
        except ValueError:
            values['ROC-AUC']=None
            result['warnings'].append('ROC-AUC is unavailable because the held-out set does not contain every class.')
        matrix=metrics.confusion_matrix(y,predicted,labels=labels)
        result['tables'].append(table('Confusion matrix (default decision rule)',[{'Actual':c,**{f'Predicted [{j+1}] {d}':int(matrix[i,j]) for j,d in enumerate(classes)}} for i,c in enumerate(classes)]))
        fig=go.Figure(go.Heatmap(z=matrix.tolist(),x=[html.escape(c) for c in classes],y=[html.escape(c) for c in classes],colorscale='Greens',text=matrix.tolist(),texttemplate='%{text}'))
        fig.update_xaxes(title='Predicted');fig.update_yaxes(title='Actual')
        result['figures'].append(figure('Held-out confusion matrix',fig))
        if len(classes)==2:
            result['thresholds']=[threshold_counts(y,probs[:,1],t/100) for t in range(101)]
            result['threshold_classes']=classes
            result['notes'].append(f'Binary positive class: {classes[1]}. The live threshold panel recalculates held-out counts at cutoffs 0.00–1.00. Choosing a cutoff using this test set makes subsequent test metrics exploratory; validate that cutoff on fresh data.')
            if len(np.unique(y))==2:
                fpr,tpr,_=metrics.roc_curve(y,probs[:,1])
                result['figures'].append(figure('ROC curve',go.Figure(go.Scatter(x=fpr.tolist(),y=tpr.tolist(),mode='lines'))))
        else:result['notes'].append('Multiclass precision, recall, F1, and one-vs-rest ROC-AUC use support-weighted averaging. Threshold adjustment is available for binary classification.')
    result['metrics'].extend({'name':name,'value':number(value)} for name,value in values.items())
    result['notes'].append('All reported predictive metrics use the held-out test set. Plot previews show at most 2,000 test observations. The downloaded pipeline retains the training-only fit; it is not refitted on the test set.')
    # Original-column permutation importance works consistently for every estimator,
    # including pipelines with one-hot encoding and polynomial expansion.
    importance_x=x.iloc[:min(500,len(x))]
    importance_y=np.asarray(y)[:len(importance_x)]
    scoring='accuracy' if classes is not None else 'neg_mean_absolute_error'
    importance=permutation_importance(pipeline,importance_x,importance_y,n_repeats=3,random_state=seed,n_jobs=1,scoring=scoring)
    rows=[{'Feature':c,'Importance':number(mean),'Std. dev.':number(std)} for c,mean,std in zip(x.columns,importance.importances_mean,importance.importances_std)]
    rows.sort(key=lambda r:r['Importance'] or 0,reverse=True)
    result['tables'].append(table('Permutation feature importance',rows))
    result['figures'].append(figure('Permutation feature importance (top 30)',go.Figure(go.Bar(x=[r['Importance'] for r in rows[:30]],y=[html.escape(r['Feature']) for r in rows[:30]],orientation='h'))))
    result['notes'].append(f'Permutation importance uses {len(importance_x)} held-out rows, three repeats, and {scoring}. It measures score reduction after shuffling an original feature; negative values are possible. Correlated features can share or mask importance.')


def projection_plot(result, coordinates, labels=None, title='Embedding'):
    points=np.asarray(coordinates)
    indices=np.linspace(0,len(points)-1,min(3000,len(points)),dtype=int)
    points=points[indices]
    fig=go.Figure()
    if labels is None:groups=[('Observations',np.ones(len(points),dtype=bool))]
    else:
        labels=np.asarray(labels)[indices]
        unique,counts=np.unique(labels,return_counts=True)
        shown=unique[np.argsort(counts)[-30:]] if len(unique)>30 else unique
        groups=[('Noise' if label==-1 else f'Cluster {label}',labels==label) for label in shown]
        if len(unique)>30:
            groups.append(('Other clusters',~np.isin(labels,shown)))
            result['notes'].append('The cluster preview groups clusters beyond the 30 largest into Other clusters. Downloaded assignments preserve every cluster ID.')
    for name,mask in groups:
        fig.add_trace(go.Scatter(x=points[mask,0].tolist(),y=points[mask,1].tolist(),mode='markers',name=name,marker={'size':5,'opacity':.7}))
    fig.update_xaxes(title='Component 1');fig.update_yaxes(title='Component 2')
    result['figures'].append(figure(title,fig))
    result['notes'].append('Projection previews show at most 3,000 observations and the first two components. Download the complete assignments/embedding as CSV.')
