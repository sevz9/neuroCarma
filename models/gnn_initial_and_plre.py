
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.nn.pytorch as dglnn

from torch import Tensor
from torch.nn.parameter import Parameter


normalisation_name_to_class = {
    'batch': nn.BatchNorm1d,
    'layer': nn.LayerNorm,
}

activation_name_to_class = {
    'gelu': nn.GELU,
    'none': nn.Identity,
}

convolution_name_to_class = {
    'sage': dglnn.SAGEConv
}

class PeriodicEmbeddings(nn.Module):
    def __init__(
        self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


class NLinear(nn.Module):
    def __init__(
        self, n_features: int, d_in: int, d_out: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_in, d_out))
        self.bias = Parameter(Tensor(n_features, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n_features):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class PLREmbeddings(nn.Sequential):
    """The PLR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'.

    Additionally, the 'lite' option is added. Setting it to `False` gives you the original PLR
    embedding from the above paper. We noticed that `lite=True` makes the embeddings
    noticeably more lightweight without critical performance loss, and we used that for our model.
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
        lite: bool,
    ) -> None:
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            (
                nn.Linear(2 * n_frequencies, d_embedding)
                if lite
                else NLinear(n_features, 2 * n_frequencies, d_embedding)
            ),
            nn.ReLU(),
        )

    

class LinearLayer(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        normalisation_name, 
        activation_name, 
        apply_skip_connection=False,
    ):
        super().__init__()
        self.transformation = nn.Linear(in_features, out_features)
        self.normalisation = normalisation_name_to_class[normalisation_name](in_features)
        self.activation = activation_name_to_class[activation_name]()
        self.apply_skip_connection = apply_skip_connection

    def forward(self, features):
        out = self.normalisation(features)
        return (
            out + self.activation(self.transformation(out))
            if self.apply_skip_connection else
            self.activation(self.transformation(out))
        )
    

class GraphConvolutionLayer(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        normalisation_name, 
        convolution_name, 
        convolution_params, 
        activation_name, 
        apply_skip_connection=False,
    ):
        super().__init__()
        self.convolution = convolution_name_to_class[convolution_name](in_features, out_features, **convolution_params)
        self.normalisation = normalisation_name_to_class[normalisation_name](in_features)
        self.activation = activation_name_to_class[activation_name]()
        self.apply_skip_connection = apply_skip_connection

    def forward(self, graph, features):
        
        out = self.normalisation(features)
        conv = self.convolution(graph, out)
        
        num_dst_nodes = conv.shape[0]
        
        # assert torch.all(torch.arange(num_dst_nodes) == graph.dstnodes().cpu())
        
        # try:
        if self.apply_skip_connection:
            out = conv + out[:num_dst_nodes, ...] # apply skip connection only to dst nodes
        else:
            out  = conv
        
        x = self.activation(out)
        # except RuntimeError:
        #     breakpoint()
        
        return x
        

class GraphNeuralNetwork(nn.Module):
    def __init__(
        self, 
        num_input_features, 
        num_hidden_features, 
        normalisation_name,
        convolution_name,
        convolution_params,
        activation_name,
        apply_skip_connection,
        num_preprocessing_layers, 
        num_encoder_layers, 
        num_predictor_layers,
        **kwargs,
        
    ):
        super().__init__()


        preprocessing_feature_list = [num_input_features] + [num_hidden_features] * num_preprocessing_layers
        encoder_feature_list = [num_hidden_features] * (num_encoder_layers + 1)
        
        
        predictor_feature_list = [num_hidden_features] * num_predictor_layers + [1]
        
        self.apply_skip_connection = apply_skip_connection
        self.preprocessing = nn.Sequential(*[
            LinearLayer(
                preprocessing_feature_list[idx], 
                preprocessing_feature_list[idx + 1], 
                normalisation_name, 
                activation_name, 
                apply_skip_connection if idx != 0 else False
            ) for idx in range(len(preprocessing_feature_list) - 1)
        ])
        self.encoder = nn.Sequential(*[
            GraphConvolutionLayer(
                encoder_feature_list[idx], 
                encoder_feature_list[idx + 1], 
                normalisation_name, 
                convolution_name, 
                convolution_params, 
                activation_name, 
                apply_skip_connection
            ) for idx in range(len(encoder_feature_list) - 1)
        ])
        self.predictor = nn.Sequential(*[
            LinearLayer(
                predictor_feature_list[idx], 
                predictor_feature_list[idx + 1], 
                normalisation_name, 
                activation_name if idx + 1 != len(predictor_feature_list) - 1 else 'none', 
                apply_skip_connection if idx + 1 != len(predictor_feature_list) - 1 else False
            ) for idx in range(len(predictor_feature_list) - 1)
        ])

    def forward(self, message_flow_graphs, features, *args, **kwargs):
        out = self.preprocessing(features)
        for convolution, mfg in zip(self.encoder, message_flow_graphs):
            out = convolution(mfg, out)
            
        out = self.predictor(out)
        return out



class GNNWithPLREmbeddings(nn.Module):
    
    def __init__(
        self, 
        num_input_features, 
        num_hidden_features, 
        normalisation_name,
        convolution_name,
        convolution_params,
        activation_name,
        apply_skip_connection,
        num_preprocessing_layers, 
        num_encoder_layers, 
        num_predictor_layers,
        
        n_frequencies: int=48,
        frequency_scale: float=0.01,
        d_embedding: int=16,
        lite: bool=True,
        
        
    ):
        super().__init__()
        
        self.apply_skip_connection = apply_skip_connection
        
        
        self.features_encoder = PLREmbeddings(
            n_features=num_input_features,
            n_frequencies=n_frequencies,
            frequency_scale=frequency_scale,
            d_embedding=d_embedding,
            lite=lite
        )
        
        self.projection = nn.Linear(
                            num_input_features * d_embedding, 
                            num_hidden_features,
                        )
        
        
        preprocessing_feature_list = [num_hidden_features] * num_preprocessing_layers
        encoder_feature_list = [num_hidden_features] * (num_encoder_layers + 1)
        predictor_feature_list = [num_hidden_features] * num_predictor_layers + [1]
        
        
        self.preprocessing = nn.Sequential(*[
            LinearLayer(
                preprocessing_feature_list[idx], 
                preprocessing_feature_list[idx + 1], 
                normalisation_name, 
                activation_name, 
                apply_skip_connection if idx != 0 else False
            ) for idx in range(len(preprocessing_feature_list) - 1)
        ])
        
        
        self.encoder = nn.Sequential(*[
            GraphConvolutionLayer(
                encoder_feature_list[idx], 
                encoder_feature_list[idx + 1], 
                normalisation_name, 
                convolution_name, 
                convolution_params, 
                activation_name, 
                apply_skip_connection=apply_skip_connection
            ) for idx in range(len(encoder_feature_list) - 1)
        ])
        
        
        self.predictor = nn.Sequential(*[
            LinearLayer(
                predictor_feature_list[idx], 
                predictor_feature_list[idx + 1], 
                normalisation_name, 
                activation_name if idx + 1 != len(predictor_feature_list) - 1 else 'none', 
                apply_skip_connection if idx + 1 != len(predictor_feature_list) - 1 else False
            ) for idx in range(len(predictor_feature_list) - 1)
        ])

    def forward(self, message_flow_graphs, features, *args, **kwargs):
        
        features_transformed = self.features_encoder(features).view(features.shape[0], -1)
        features_transformed = F.relu(self.projection(features_transformed))
        
        out = self.preprocessing(features_transformed)
        
        for convolution, mfg in zip(self.encoder, message_flow_graphs):
            out = convolution(mfg, out)

        out = self.predictor(out)
        
        return out

model_name_to_class = {
    'GNN': GraphNeuralNetwork,
    'none': None,
    "GNN-PLRE": GNNWithPLREmbeddings,
}


def create_graph_model(model_name, model_params):
    model_class = model_name_to_class[model_name]
    model = model_class(**model_params)

    return model
