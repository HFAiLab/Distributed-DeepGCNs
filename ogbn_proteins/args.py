import argparse


def get_args():
    parser = argparse.ArgumentParser(description='DeeperGCN')
    # dataset
    parser.add_argument('--dataset', type=str, default='ogbn-proteins', help='dataset name (default: ogbn-proteins)')
    parser.add_argument('--cluster_number', type=int, default=10, help='the number of sub-graphs for training')
    parser.add_argument('--valid_cluster_number', type=int, default=5, help='the number of sub-graphs for evaluation')
    parser.add_argument('--aggr', type=str, default='add', help='the aggregation operator to obtain nodes\' initial features [mean, max, add]')
    parser.add_argument('--nf_path', type=str, default='init_node_features_add.pt', help='the file path of extracted node features saved.')
    # training & eval settings
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 100)')
    parser.add_argument('--num_evals', type=int, default=1, help='The number of evaluation times')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.0)
    # model
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of the networks')
    parser.add_argument('--mlp_layers', type=int, default=2, help='the number of layers of mlp in conv')
    parser.add_argument('--hidden_channels', type=int, default=64, help='the dimension of embeddings of nodes and edges')
    parser.add_argument('--block', default='plain', type=str, help='graph backbone block type {res+, res, dense, plain}')
    parser.add_argument('--conv', type=str, default='gen', help='the type of GCNs')
    parser.add_argument('--gcn_aggr', type=str, default='max', help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, softmax_sum, power, power_sum]')
    parser.add_argument('--norm', type=str, default='layer', help='the type of normalization layer')
    parser.add_argument('--num_tasks', type=int, default=1, help='the number of prediction tasks')
    # learnable parameters
    parser.add_argument('--t', type=float, default=1.0, help='the temperature of SoftMax')
    parser.add_argument('--p', type=float, default=1.0, help='the power of PowerMean')
    parser.add_argument('--y', type=float, default=0.0, help='the power of degrees')
    parser.add_argument('--learn_t', action='store_true')
    parser.add_argument('--learn_p', action='store_true')
    parser.add_argument('--learn_y', action='store_true')
    # message norm
    parser.add_argument('--msg_norm', action='store_true')
    parser.add_argument('--learn_msg_scale', action='store_true')
    # encode edge in conv
    parser.add_argument('--conv_encode_edge', action='store_true')
    # if use one-hot-encoding node feature
    parser.add_argument('--use_one_hot_encoding', action='store_true')

    return parser.parse_args()
