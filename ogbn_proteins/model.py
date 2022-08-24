
import torch
from gcn_lib.sparse.torch_vertex import GENConv
from gcn_lib.sparse.torch_nn import norm_layer
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block

        self.checkpoint_grad = False

        hidden_channels = args.hidden_channels
        num_tasks = args.num_tasks
        conv = args.conv
        aggr = args.gcn_aggr

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        y = args.y
        self.learn_y = args.learn_y

        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        conv_encode_edge = args.conv_encode_edge
        norm = args.norm
        mlp_layers = args.mlp_layers
        node_features_file_path = args.nf_path

        self.use_one_hot_encoding = args.use_one_hot_encoding

        # save gpu mem using gradient ckpt
        if aggr not in ['add', 'max', 'mean'] and self.num_layers > 15:
            self.checkpoint_grad = True
            self.ckp_k = 9

        # if self.block == 'res+':
        #     print('LN/BN->ReLU->GraphConv->Res')
        # elif self.block == 'res':
        #     print('GraphConv->LN/BN->ReLU->Res')
        # elif self.block == 'dense':
        #     raise NotImplementedError('To be implemented')
        # elif self.block == "plain":
        #     print('GraphConv->LN/BN->ReLU')
        # else:
        #     raise Exception('Unknown block Type')
        assert self.block in ['res+', 'res', 'dense', 'plain']

        self.gcns = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              y=y, learn_y=self.learn_y,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=conv_encode_edge, edge_feat_dim=hidden_channels,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.layer_norms.append(norm_layer(norm, hidden_channels))

        self.node_features = torch.load(node_features_file_path).cuda(non_blocking=True)

        if self.use_one_hot_encoding:
            self.node_one_hot_encoder = torch.nn.Linear(8, 8)
            self.node_features_encoder = torch.nn.Linear(8 * 2, hidden_channels)
        else:
            self.node_features_encoder = torch.nn.Linear(8, hidden_channels)

        self.edge_encoder = torch.nn.Linear(8, hidden_channels)

        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, x, node_index, edge_index, edge_attr):

        node_features_1st = self.node_features[node_index]

        if self.use_one_hot_encoding:
            node_features_2nd = self.node_one_hot_encoder(x)
            # concatenate
            node_features = torch.cat((node_features_1st, node_features_2nd), dim=1)
        else:
            node_features = node_features_1st

        h = self.node_features_encoder(node_features)

        edge_emb = self.edge_encoder(edge_attr)

        if self.block == 'res+':
            h = self.gcns[0](h, edge_index, edge_emb)

            if self.checkpoint_grad:
                for layer in range(1, self.num_layers):
                    h1 = self.layer_norms[layer-1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    if layer % self.ckp_k != 0:
                        res = checkpoint(self.gcns[layer], h2, edge_index, edge_emb)
                        h = res + h
                    else:
                        h = self.gcns[layer](h2, edge_index, edge_emb) + h

            else:
                for layer in range(1, self.num_layers):
                    h1 = self.layer_norms[layer-1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    h = self.gcns[layer](h2, edge_index, edge_emb) + h

            h = F.relu(self.layer_norms[self.num_layers-1](h))
            h = F.dropout(h, p=self.dropout, training=self.training)

            return self.node_pred_linear(h)

        elif self.block == 'res':

            h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.layer_norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

            return self.node_pred_linear(h)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.layer_norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.layer_norms[layer](h1)
                h = F.relu(h2)
                h = F.dropout(h, p=self.dropout, training=self.training)

            return self.node_pred_linear(h)

        else:
            raise Exception('Unknown block Type')
