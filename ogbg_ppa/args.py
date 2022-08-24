import argparse


def get_args():

    parser = argparse.ArgumentParser(description='DeeperGCN')
    # dataset
    parser.add_argument('--dataset', type=str, default="ogbg-ppa", help='dataset name (default: ogbg-ppa)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    # extract node features
    parser.add_argument('--aggr', type=str, default='add', help='the aggregation operator to obtain nodes\' initial features [mean, max, add]')
    # training & eval settings
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.5)
    # model
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of the networks')
    parser.add_argument('--mlp_layers', type=int, default=2, help='the number of layers of mlp in conv')
    parser.add_argument('--hidden_channels', type=int, default=128, help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--block', default='res+', type=str, help='graph backbone block type {res+, res, dense, plain}')
    parser.add_argument('--conv', type=str, default='gen', help='the type of GCNs')
    parser.add_argument('--gcn_aggr', type=str, default='max', help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
    parser.add_argument('--norm', type=str, default='layer', help='the type of normalization layer')
    parser.add_argument('--num_tasks', type=int, default=1, help='the number of prediction tasks')
    # learnable parameters
    parser.add_argument('--t', type=float, default=1.0, help='the temperature of SoftMax')
    parser.add_argument('--p', type=float, default=1.0, help='the power of PowerMean')
    parser.add_argument('--learn_t', action='store_true')
    parser.add_argument('--learn_p', action='store_true')
    # message norm
    parser.add_argument('--msg_norm', action='store_true')
    parser.add_argument('--learn_msg_scale', action='store_true')
    # encode edge in conv
    parser.add_argument('--conv_encode_edge', action='store_true')
    # graph pooling type
    parser.add_argument('--graph_pooling', type=str, default='mean', help='graph pooling method')
    # others, eval steps
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--num_layers_threshold', type=int, default=14)

    return parser.parse_args()