import torch
from utils.sparse_softmax import Sparsemax
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.utils import softmax, dense_to_sparse, to_dense_adj
from torch_geometric import loader
import torch.nn.functional as F


MAX_LOGSTD = 10


class Edgelayer(torch.nn.Module):
    def __init__(self, in_channels, sparse=True, negative_slop=0.2):
        super(Edgelayer, self).__init__()
        self.in_channels = in_channels
        self.negative_slop = negative_slop
        self.sparse = sparse
        self.att = torch.nn.Parameter(torch.Tensor(1, self.in_channels * 2))
        torch.nn.init.xavier_uniform_(self.att.data)
        self.sparse_attention = Sparsemax()

    def forward(self, x, edge_index):
        row, col = edge_index  # inital row col
        weights = (torch.cat([x[row], x[col]], dim=1) * self.att).sum(dim=-1)
        weights = torch.nn.functional.leaky_relu(weights, self.negative_slop)
        if self.sparse:
            new_edge_attr = self.sparse_attention(weights, row)
        else:
            new_edge_attr = softmax(weights, row, x.size(0))
        ind = torch.where(new_edge_attr != 0)[0]
        new_edge_index = edge_index[:, ind]
        new_edge_attr = new_edge_attr[ind]
        return new_edge_index, new_edge_attr


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels)
        self.conv2 = GATConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index).sigmoid()


class BatchInProDecoder(torch.nn.Module):
    def __init__(self, z_features, x_features, dec_layer=2):
        super(BatchInProDecoder, self).__init__()
        self.decoder = torch.nn.Sequential()
        self.dec_layer = dec_layer
        for i in range(dec_layer):
            self.decoder.add_module(f'enc_layer_{i}', GCNConv(z_features, 2 * x_features) if i == 0
            else GCNConv(2 * x_features, 2 * x_features))
        self.lin = torch.nn.Linear(2 * x_features, x_features, bias=True)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, z_, edge_index, edge_attr, sigmoid=True):
        adj = torch.matmul(z_, z_.T).sigmoid() if sigmoid else torch.matmul(z_, z_.T)
        new_edge_index, new_edge_attr = dense_to_sparse(adj)
        for layer in range(self.dec_layer):
            z_ = self.decoder[layer]((z_), edge_index, edge_attr).relu()
            z_ = self.dropout(z_)
        z_ = self.lin(z_).sigmoid()
        return z_, new_edge_index


class GVAE(torch.nn.Module):
    def __init__(self, in_features, z_features, enc_layer=1):
        super(GVAE, self).__init__()
        self.dense = torch.nn.Linear(in_features, 2*z_features)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.encoder = torch.nn.Sequential()
        for i in range(enc_layer):
            self.encoder.add_module(f'enc_layer_{i}', GCNConv(2*z_features, 2*z_features))
        self.conv_mu = GCNConv(2*z_features, z_features)
        self.conv_logstd = GCNConv(2*z_features, z_features)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu=None, logstd=None):
        logstd = logstd.clamp(max=MAX_LOGSTD)
        return - 0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def forward(self, x, edge_index, edge_attr):
        x = self.dense(x).relu()
        x = self.dropout(x)
        for layer in self.encoder[:-1]:
            x = layer(x, edge_index, edge_attr)
            x = self.dropout(x)
        self.mu = self.conv_mu(x, edge_index, edge_attr)
        self.logstd = self.conv_logstd(x, edge_index, edge_attr)
        return self.reparametrize(self.mu, self.logstd), self.mu, self.logstd, self.kl_loss(self.mu, self.logstd)


class GVAD(torch.nn.Module):
    def __init__(self, x_features, z_features, enc_layer=1, dec_layer=1):
        super(GVAD, self).__init__()
        self.edge_layer = Edgelayer(x_features)
        self.enc = GVAE(x_features, z_features, enc_layer)
        self.dec = BatchInProDecoder(z_features, x_features, dec_layer)
        self.name = 'GVAD'

    def forward(self, batch_data, sigmoid=True):
        graph_list = batch_data.to_data_list()
        for graph in graph_list:
            graph.edge_index, graph.edge_attr = self.edge_layer(graph.x, graph.edge_index)
            graph.z, graph.mu, graph.logstd, graph.kl_loss = self.enc(graph.x, graph.edge_index, graph.edge_attr)
            graph.x_, graph.edge_index_ = self.dec(graph.z, graph.edge_index, graph.edge_attr)
        return Batch.from_data_list(graph_list)