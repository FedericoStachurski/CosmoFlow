import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from MultiLayerPerceptron.utilities import get_script_path
import dill as pickle
from pathlib import Path


class LinearModel(nn.Module):
    def __init__(self, in_features, out_features, neurons, n_layers, activation, name, out_activation=None, initialisation=xavier_uniform_, use_dropout=False,drop_p=0.25, use_bn=False):
        super().__init__()
        self.initial = initialisation
        self.name = name

        layers = [nn.Linear(in_features, neurons[0]), activation()]
        for i in range(n_layers - 1):
            layers.append(nn.Linear(neurons[i], neurons[i + 1]))

            if use_dropout:
                layers.append(nn.Dropout(drop_p))  
            layers.append(activation())
            if use_bn:
                layers.append(nn.BatchNorm1d(num_features=neurons[i+1]))
                
        layers.append(nn.Linear(neurons[-1], out_features))
        if out_activation is not None:
            layers.append(out_activation())
        
        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.init_weights)

    def forward(self, x):
        return self.layers(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.initial(m.weight)


def create_mlp(input_features, output_features, neurons, layers, activation, model_name, out_activation=None, init=xavier_uniform_, device=None, norm_type='z-score', use_dropout=False,drop_p=0.25, use_bn=False):
    if isinstance(neurons, list):
        if len(neurons) != layers:
            raise RuntimeError('Length of neuron vector does not equal number of hidden layers.')
    else:
        neurons = [neurons, ]
    model = LinearModel(input_features, output_features, neurons, layers, activation, model_name, initialisation=init, use_dropout=use_dropout,drop_p=drop_p,use_bn=use_bn, out_activation=out_activation)
    model.norm_type=norm_type
    Path(get_script_path()+f'/models/{model_name}/').mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(get_script_path()+f'/models/{model_name}/function.pickle', "wb"), pickle.HIGHEST_PROTOCOL)  # save blank model

    if device is not None:
        model = model.to(device)
    return model


def load_mlp(model_name, device, get_state_dict=False):
    model = pickle.load(open(get_script_path()+f'/models/{model_name}/function.pickle', "rb"))  # load blank model
    if get_state_dict:
        model.load_state_dict(torch.load(open(get_script_path()+f'/models/{model_name}/model.pth', "rb"), map_location=device))
    return model
