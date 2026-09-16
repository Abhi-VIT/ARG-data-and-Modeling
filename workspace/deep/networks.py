import torch
from torch import nn


def activation(name):return {'relu':nn.ReLU,'tanh':nn.Tanh,'gelu':nn.GELU}[name]()


def dense(widths, name, dropout, final_activation=False):
    layers=[]
    for i,(a,b) in enumerate(zip(widths,widths[1:])):
        layers.append(nn.Linear(a,b))
        if i<len(widths)-2 or final_activation:
            layers.extend([activation(name),nn.Dropout(dropout)])
    return nn.Sequential(*layers)


class SequenceNetwork(nn.Module):
    def __init__(self,kind,inputs,outputs,c):
        super().__init__()
        klass={'rnn':nn.RNN,'lstm':nn.LSTM,'gru':nn.GRU}[kind]
        options={'nonlinearity':'relu' if c['activation']=='relu' else 'tanh'} if kind=='rnn' else {}
        self.recurrent=klass(inputs,c['recurrent_units'],num_layers=c['recurrent_layers'],batch_first=True,
                            dropout=c['dropout'] if c['recurrent_layers']>1 else 0,**options)
        self.head=dense([c['recurrent_units'],*c['hidden_layers'],outputs],c['activation'],c['dropout'])

    def forward(self,x):
        values,_=self.recurrent(x)
        return self.head(values[:,-1,:])


class Autoencoder(nn.Module):
    def __init__(self,inputs,c):
        super().__init__()
        self.encoder=dense([inputs,*c['hidden_layers'],c['latent_dim']],c['activation'],c['dropout'])
        self.decoder=dense([c['latent_dim'],*reversed(c['hidden_layers']),inputs],c['activation'],c['dropout'])

    def forward(self,x):return self.decoder(self.encoder(x))


def build_network(config,inputs,outputs):
    kind=config['architecture']
    if kind=='mlp':return dense([inputs,*config['hidden_layers'],outputs],config['activation'],config['dropout'])
    if kind in {'rnn','lstm','gru'}:return SequenceNetwork(kind,inputs,outputs,config)
    if kind=='autoencoder':return Autoencoder(inputs,config)
    layers=[];previous=3
    for channels in config['conv_channels']:
        layers.extend([nn.Conv2d(previous,channels,3,padding=1),activation(config['activation']),nn.MaxPool2d(2),nn.Dropout2d(config['dropout'])])
        previous=channels
    layers.extend([nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),dense([previous,*config['hidden_layers'],outputs],config['activation'],config['dropout'])])
    return nn.Sequential(*layers)


def restore_checkpoint(path):
    """Load a server-produced checkpoint using tensor/primitive-only deserialization."""
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format_version')!=1:raise ValueError('Unsupported checkpoint format.')
    model=build_network(checkpoint['config'],checkpoint['input_width'],checkpoint['output_width'])
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model,checkpoint
