from torch_geometric.nn.models.autoencoder import ARGA, VGAE
from torch_geometric.nn import GCNConv, GATConv
import torch
import torch.nn.functional as F
from torch_geometric.utils import softmax, dense_to_sparse, to_dense_adj
from torch_geometric.data import Data, Batch
from utils.sparse_softmax import Sparsemax


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
        row, col = edge_index
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


class ARGVA(ARGA):
    def __init__(self, encoder, discriminator, decoder=None):
        super().__init__(encoder, discriminator, decoder)
        self.VGAE = VGAE(encoder, decoder)

    @property
    def __mu__(self):
        return self.VGAE.__mu__

    @property
    def __logstd__(self):
        return self.VGAE.__logstd__

    def reparametrize(self, mu, logstd):
        return self.VGAE.reparametrize(mu, logstd)

    def encode(self, *args, **kwargs):
        """"""
        return self.VGAE.encode(*args, **kwargs)

    def kl_loss(self, mu=None, logstd=None):
        return self.VGAE.kl_loss(mu, logstd)


class Encoder(torch.nn.Module):
    def __init__(self, x_features, z_features, enc_layer=1):
        super(Encoder, self).__init__()
        self.dense = torch.nn.Linear(x_features, 2 * z_features)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.encoder = torch.nn.Sequential()
        for i in range(enc_layer):
            self.encoder.add_module(f'enc_layer_{i}', GCNConv(2 * z_features, 2 * z_features))
        self.conv_mu = GCNConv(2 * z_features, z_features)
        self.conv_logstd = GCNConv(2 * z_features, z_features)

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
        return self.mu, self.logstd


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, z_features, x_features, dec_layer=2):
        super(Decoder, self).__init__()
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
        z_ = self.lin(z_)
        return z_, new_edge_index


class AGVAE(ARGVA):
    def __init__(self, x_features, z_features, enc_layer=1, dec_layer=1):
        encoder = Encoder(x_features, z_features, enc_layer)
        decoder = Decoder(z_features, x_features, dec_layer)
        discriminator = Discriminator(z_features, 2*z_features, z_features)
        super(AGVAE, self).__init__(encoder, discriminator, discriminator)
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.edge_layer = Edgelayer(x_features)
        self.name = 'AGVAE'


    def forward(self, batch_data, sigmoid=True):
        graph_list = batch_data.to_data_list()
        for graph in graph_list:
            graph.edge_index, graph.edge_attr = self.edge_layer(graph.x, graph.edge_index)
            graph.z = self.encode(graph.x, graph.edge_index, graph.edge_attr)
            graph.mu, graph.logstd = self.__mu__, self.__logstd__
            graph.kl_loss = self.VGAE.kl_loss(graph.mu, graph.logstd)
            graph.x_, graph.edge_index_ = self.decoder(graph.z, graph.edge_index, graph.edge_attr)
        return Batch.from_data_list(graph_list)


