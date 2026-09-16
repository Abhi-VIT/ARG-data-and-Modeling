from ..ingestion import DataError
from ..ml.training import bounded

ARCHITECTURES=['mlp','cnn','rnn','lstm','gru','autoencoder']
DEFAULTS={'architecture':'mlp','task':'regression','hidden_layers':[64,32],'activation':'relu',
          'optimizer':'adam','learning_rate':.001,'batch_size':32,'epochs':30,'dropout':.1,
          'early_stopping':True,'patience':5,'validation_size':.2,'test_size':.2,'seed':42,
          'device':'auto','sequence_length':10,'recurrent_layers':1,'recurrent_units':32,
          'conv_channels':[16,32],'latent_dim':2,'anomaly_quantile':.95}


def configuration(options):
    if not isinstance(options,dict):raise DataError('Provide deep-learning settings.')
    c={**DEFAULTS,**options}
    if c['architecture'] not in ARCHITECTURES:raise DataError('Choose a supported neural architecture.')
    if c['architecture']=='cnn':c['task']='classification'
    if c['architecture']=='autoencoder':c['task']='reconstruction'
    if c['task'] not in {'regression','classification','reconstruction'} or (c['task']=='reconstruction' and c['architecture']!='autoencoder'):
        raise DataError('Choose regression or classification, or use an autoencoder for reconstruction.')
    for field,allowed in {'activation':['relu','tanh','gelu'],'optimizer':['adam','sgd','rmsprop'],'device':['auto','cpu','cuda']}.items():
        if c[field] not in allowed:raise DataError('Choose a listed '+field+'.')
    for field,lo,hi in [('batch_size',4,256),('epochs',1,200),('patience',1,30),('seed',0,2**31-1),
                        ('sequence_length',2,100),('recurrent_layers',1,3),('recurrent_units',4,256),('latent_dim',1,32)]:
        c[field]=bounded(c[field],lo,hi,field,True)
    for field,lo,hi in [('learning_rate',.00001,.1),('dropout',0,.7),('validation_size',.1,.3),('test_size',.1,.3),('anomaly_quantile',.8,.999)]:
        c[field]=bounded(c[field],lo,hi,field)
    if not isinstance(c['early_stopping'],bool):raise DataError('Early stopping must be enabled or disabled.')
    for field,lo,hi in [('hidden_layers',4,512),('conv_channels',4,128)]:
        if not isinstance(c[field],list) or not 1<=len(c[field])<=4:raise DataError(field+' must contain 1–4 layer widths.')
        c[field]=[bounded(v,lo,hi,field,True) for v in c[field]]
    return c
