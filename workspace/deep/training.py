import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from torch import nn
from sklearn import metrics
from ..ingestion import DataError
from ..analytics.common import report, table, figure, number, clean_json
from .config import configuration
from .data import tabular_data, split_indices
from .networks import build_network


def choose_device(requested):
    available=torch.cuda.is_available()
    if requested=='cuda' and not available:raise DataError('CUDA was requested but this worker has no available CUDA device. Choose CPU/Auto or start a CUDA-enabled deep worker.')
    return torch.device('cuda' if available and requested!='cpu' else 'cpu')


def outputs(model,x,device,batch_size):
    chunks=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(x),batch_size):
            batch=torch.as_tensor(x[start:start+batch_size],dtype=torch.float32,device=device)
            chunks.append(model(batch).cpu().numpy())
    return np.concatenate(chunks)


def atomic_checkpoint(path,packet):
    temporary=Path(str(path)+'.tmp')
    try:
        torch.save(packet,temporary)
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def train_deep(options,frame=None,image_data=None,checkpoint_path=None,on_epoch=lambda data:None,source=None):
    c=configuration(options);device=choose_device(c['device']);torch.set_num_threads(2)
    torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    if device.type=='cuda':torch.cuda.manual_seed_all(c['seed'])
    if c['architecture']=='cnn':
        if image_data is None:raise DataError('Select a validated image collection for CNN training.')
        pixels,labels,manifest=image_data
        indices=split_indices(len(labels),c,labels)
        if any(len(ids)<2 for ids in indices):raise DataError('Each image split needs at least two images.')
        datasets=[(pixels[ids].astype(np.float32)/255.,labels[ids]) for ids in indices]
        meta={'classes':manifest['classes'],'input_width':3,'output_width':len(manifest['classes']),
              'features':[],'scaler':None,'target_scaler':None,'row_ids':[(ids+1).tolist() for ids in indices],
              'notes':['Images were deduplicated, center-cropped/resized to RGB 64×64, and scaled to [0,1]. Stratified train/validation/test splits use the selected seed.'],
              'image_size':64}
    else:
        if frame is None:raise DataError('Import a tabular dataset first.')
        datasets,meta=tabular_data(frame,c)
    if c['epochs']*len(datasets[0][0])>2_000_000:raise DataError('Training exceeds two million row/window visits. Reduce epochs or rows.')
    model=build_network(c,meta['input_width'],meta['output_width']).to(device)
    count=sum(p.numel() for p in model.parameters())
    if count>2_000_000:raise DataError('Network exceeds two million parameters. Reduce layer widths or depth.')
    optimizer={'adam':torch.optim.Adam,'sgd':torch.optim.SGD,'rmsprop':torch.optim.RMSprop}[c['optimizer']](model.parameters(),lr=c['learning_rate'])
    criterion=nn.CrossEntropyLoss() if c['task']=='classification' else nn.MSELoss()
    (xtrain,ytrain),(xval,yval),(xtest,ytest)=datasets
    history=[];best_loss=float('inf');best_state=None;best_epoch=0;stale=0
    rng=np.random.default_rng(c['seed']);deadline=time.monotonic()+900
    packet={'format_version':1,'config':c,'input_width':meta['input_width'],'output_width':meta['output_width'],
            'preprocessing':{k:v for k,v in meta.items() if k not in {'row_ids','notes'}},'source':source or {},'torch_version':str(torch.__version__)}
    def evaluate_loss(logits,truth):
        target=torch.as_tensor(truth,dtype=torch.long if c['task']=='classification' else torch.float32)
        if c['task']=='regression':target=target.reshape(-1,1)
        return float(criterion(torch.as_tensor(logits),target).item())
    for epoch in range(1,c['epochs']+1):
        model.train();total_loss=0.;correct=0
        for ids in np.array_split(rng.permutation(len(xtrain)),max(1,int(np.ceil(len(xtrain)/c['batch_size'])))):
            if time.monotonic()>deadline:raise DataError('Training reached the 15-minute limit. The best completed checkpoint is retained. Reduce epochs or architecture size.')
            x=torch.as_tensor(xtrain[ids],dtype=torch.float32,device=device)
            y=torch.as_tensor(ytrain[ids],dtype=torch.long if c['task']=='classification' else torch.float32,device=device)
            if c['task']=='regression':y=y.reshape(-1,1)
            optimizer.zero_grad(set_to_none=True);logits=model(x);loss=criterion(logits,y)
            if not torch.isfinite(loss):raise DataError('Training diverged. Lower the learning rate or check extreme values.')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total_loss+=float(loss.item())*len(ids)
            if c['task']=='classification':correct+=int((logits.argmax(dim=1)==y).sum().item())
        predicted=outputs(model,xval,device,c['batch_size']);val_loss=evaluate_loss(predicted,yval)
        if not np.isfinite(val_loss):raise DataError('Validation loss is non-finite. Lower the learning rate.')
        row={'epoch':epoch,'train_loss':total_loss/len(xtrain),'validation_loss':val_loss}
        if c['task']=='classification':row.update(train_accuracy=correct/len(xtrain),validation_accuracy=float((predicted.argmax(axis=1)==yval).mean()))
        history.append(row)
        if val_loss<best_loss-1e-6:
            best_loss=val_loss;best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            packet.update(state_dict=best_state,best_epoch=best_epoch,history=history.copy())
            if checkpoint_path:atomic_checkpoint(checkpoint_path,packet)
        else:stale+=1
        on_epoch({'history':history.copy(),'epoch':epoch,'epochs':c['epochs'],'best_epoch':best_epoch,'device':str(device),
                  'checkpoint_ready':checkpoint_path is not None,'early_stopped':c['early_stopping'] and stale>=c['patience']})
        if c['early_stopping'] and stale>=c['patience']:break
    model.load_state_dict(best_state);model.eval()
    predicted=outputs(model,xtest,device,c['batch_size'])
    result=report(c['architecture'].upper()+' · '+c['task'],'deep')
    result['parameters']=c;result['selection']={'used':meta['features'] or ['RGB images']}
    result['metrics']=[{'name':name,'value':value} for name,value in [('Training samples',len(xtrain)),('Validation samples',len(xval)),('Test samples',len(xtest)),('Epochs completed',len(history)),('Best epoch',best_epoch),('Parameters',count),('Device',str(device))]]
    result['tables'].append(table('Epoch history',history))
    for suffix,title in [('loss','Training and validation loss'),('accuracy','Training and validation accuracy')]:
        if suffix=='accuracy' and c['task']!='classification':continue
        fig=go.Figure()
        for part in ['train','validation']:
            fig.add_trace(go.Scatter(x=[r['epoch'] for r in history],y=[r[part+'_'+suffix] for r in history],mode='lines+markers',name=part))
        fig.update_xaxes(title='Epoch');fig.update_yaxes(title=suffix.capitalize())
        result['figures'].append(figure(title,fig))
    classes=meta['classes']
    output=pd.DataFrame({'row_number':meta['row_ids'][2]})
    if c['architecture']=='cnn':output['image_name']=[manifest['names'][i-1] for i in meta['row_ids'][2]]
    if c['task']=='classification':
        probabilities=torch.softmax(torch.as_tensor(predicted),dim=1).numpy();guesses=probabilities.argmax(axis=1)
        average='binary' if len(classes)==2 else 'weighted'
        values={'Test accuracy':metrics.accuracy_score(ytest,guesses),'Test precision':metrics.precision_score(ytest,guesses,average=average,zero_division=0),
                'Test recall':metrics.recall_score(ytest,guesses,average=average,zero_division=0),'Test F1':metrics.f1_score(ytest,guesses,average=average,zero_division=0)}
        try:values['Test ROC-AUC']=metrics.roc_auc_score(ytest,probabilities[:,1]) if len(classes)==2 else metrics.roc_auc_score(ytest,probabilities,multi_class='ovr',labels=np.arange(len(classes)),average='weighted')
        except ValueError:values['Test ROC-AUC']=None
        matrix=metrics.confusion_matrix(ytest,guesses,labels=np.arange(len(classes)))
        result['tables'].append(table('Test confusion matrix',[{'Actual':label,**{f'Predicted [{j+1}] {name}':int(matrix[i,j]) for j,name in enumerate(classes)}} for i,label in enumerate(classes)]))
        output['actual']=[classes[v] for v in ytest];output['prediction']=[classes[v] for v in guesses]
        for j,label in enumerate(classes):output[f'probability [{j+1}] {label}']=probabilities[:,j]
        result['notes'].append('Classification uses argmax logits. '+('Binary positive class: '+classes[1]+'.' if len(classes)==2 else 'Multiclass metrics use support-weighted averaging.'))
    elif c['task']=='regression':
        scaler=meta['target_scaler'];actual=ytest*scaler['scale']+scaler['mean'];guesses=predicted[:,0]*scaler['scale']+scaler['mean']
        values={'Test RMSE':metrics.root_mean_squared_error(actual,guesses),'Test MAE':metrics.mean_absolute_error(actual,guesses),'Test R²':metrics.r2_score(actual,guesses)}
        output['actual']=actual;output['prediction']=guesses
        result['figures'].append(figure('Test predictions',go.Figure(go.Scatter(x=actual[:2000].tolist(),y=guesses[:2000].tolist(),mode='markers'))))
    else:
        errors=((predicted-xtest)**2).mean(axis=1)
        train_errors=((outputs(model,xtrain,device,c['batch_size'])-xtrain)**2).mean(axis=1)
        cutoff=float(np.quantile(train_errors,c['anomaly_quantile']))
        with torch.no_grad():latent=np.concatenate([model.encoder(torch.as_tensor(xtest[i:i+c['batch_size']],device=device)).cpu().numpy() for i in range(0,len(xtest),c['batch_size'])])
        output['reconstruction_error']=errors;output['anomaly_flag']=errors>cutoff
        for j in range(latent.shape[1]):output[f'latent_{j+1}']=latent[:,j]
        values={'Test reconstruction MSE':float(errors.mean()),'Training error cutoff':cutoff,'Test rows flagged':int((errors>cutoff).sum())}
        result['figures'].append(figure('Test reconstruction errors',go.Figure(go.Histogram(x=errors.tolist()))))
        result['figures'].append(figure('Latent representation (test rows)',go.Figure(go.Scatter(x=latent[:2000,0].tolist(),y=(latent[:2000,1] if latent.shape[1]>1 else np.zeros(min(2000,len(latent)))).tolist(),mode='markers'))))
        packet['anomaly_cutoff']=cutoff
        result['notes'].append(f'Anomaly flags use the {c["anomaly_quantile"]:.1%} quantile of training reconstruction errors in standardized feature space. They are heuristic flags, not validated anomaly labels. Latent coordinates use the best checkpoint encoder.')
    result['metrics'].extend({'name':k,'value':number(v)} for k,v in values.items())
    result['notes'].extend(meta['notes'])
    result['notes'].extend(['Validation loss selects the best checkpoint and controls early stopping. Test rows are evaluated only after training. No checkpoint is refitted on validation or test rows.',
                            'Each job has a 15-minute training-loop limit. Epoch updates and the best completed checkpoint persist across page reloads.'])
    if c['architecture']!='cnn':result['notes'].append('Numeric features use training-only median imputation and standardization; regression targets use training-only standardization. Loss curves use that standardized scale, while final regression metrics use the original target units.')
    if len(history)<c['epochs']:result['notes'].append(f'Early stopping ended training after {len(history)} epochs; epoch {best_epoch} had the lowest validation loss.')
    if device.type=='cpu' and c['device']=='auto':result['notes'].append('No CUDA device was available; this run used CPU fallback.')
    if c['architecture'] in {'lstm','gru'}:result['notes'].append('Recurrent cells use PyTorch’s built-in gate activations. The chosen activation applies to the dense prediction head.')
    if c['architecture']=='rnn':result['notes'].append('RNN cells use ReLU when selected, otherwise Tanh. The dense prediction head uses the selected activation.')
    packet.update(state_dict=best_state,history=history,best_epoch=best_epoch)
    if checkpoint_path:atomic_checkpoint(checkpoint_path,packet)
    result=clean_json(result);json.dumps(result,allow_nan=False)
    return result,packet,output
