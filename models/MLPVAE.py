import torch.nn as nn
from torch_geometric.data import Batch, Data


class MLPEncoder(nn.Module):
    def __init__(self, x_features, z_features, enc_layer):
        super(MLPEncoder, self).__init__()
        self.enc = nn.ModuleList()
        self.dropout= nn.Dropout(p=0.5)
        for i in range(enc_layer):
            self.enc.append(nn.Linear(x_features, z_features)) if i==0 else \
                self.enc.append(nn.Linear(z_features, z_features))

    def forward(self, x):
        for layer in self.enc:
            x = layer(x).relu()
            x = self.dropout(x)
        return x


class MLPDecoder(nn.Module):
    def __init__(self, z_features, x_features, dec_layer):
        super(MLPDecoder, self).__init__()
        self.dec = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.5)
        for i in range(dec_layer):
            self.dec.append(nn.Linear(z_features, x_features)) if i==0 else\
                self.dec.append(nn.Linear(x_features, x_features))

    def forward(self, x):
        for layer in self.dec[:-1]:
            x = layer(x).relu()
            x = self.dropout(x)
        return self.dec[-1](x)


class MLPAE(nn.Module):
    def __init__(self, x_features, z_features, enc_layer, dec_layer):
        super(MLPAE, self).__init__()
        self.enc = MLPEncoder(x_features, z_features, enc_layer)
        self.dec = MLPDecoder(z_features, x_features, dec_layer)

    def forward(self, batch_data):
        graph_list = batch_data.to_data_list()
        new_graph_list = []
        for graph in graph_list:
            graph.z = self.enc(graph.x)
            graph.x_ = self.dec(graph.z)
            new_graph_list.append(graph)
        return Batch.from_data_list(new_graph_list)