import argparse


def get_args():
    parser = argparse.ArgumentParser(description='DeeperGCN')
    # dataset
    parser.add_argument('--dataset', type=str, default='ogbl-collab', help='dataset name (default: ogbl-collab)')
    parser.add_argument('--self_loop', action='store_true')
    # training & eval settings
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train (default: 400)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate set for optimizer.')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024, help='the number of edges per batch')
    # model
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of the networks')
    parser.add_argument('--lp_num_layers', type=int, default=3, help='the number of layers of the link predictor model')
    parser.add_argument('--mlp_layers', type=int, default=1, help='the number of layers of mlp in conv')
    parser.add_argument('--in_channels', type=int, default=128, help='the dimension of initial embeddings of nodes')
    parser.add_argument('--hidden_channels', type=int, default=128, help='the dimension of embeddings of nodes')
    parser.add_argument('--block', default='res+', type=str, help='graph backbone block type {res+, res, dense, plain}')
    parser.add_argument('--conv', type=str, default='gen', help='the type of GCNs')
    parser.add_argument('--gcn_aggr', type=str, default='max', help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
    parser.add_argument('--norm', type=str, default='batch', help='the type of normalization layer')
    parser.add_argument('--lp_norm', type=str, default='none', help='the type of normalization layer for link predictor')
    parser.add_argument('--num_tasks', type=int, default=1, help='the number of prediction tasks')
    # learnable parameters
    parser.add_argument('--t', type=float, default=1.0, help='the temperature of SoftMax')
    parser.add_argument('--p', type=float, default=1.0, help='the power of PowerMean')
    parser.add_argument('--learn_t', action='store_true')
    parser.add_argument('--learn_p', action='store_true')
    parser.add_argument('--y', type=float, default=0.0, help='the power of softmax_sum and powermean_sum')
    parser.add_argument('--learn_y', action='store_true')

    # message norm
    parser.add_argument('--msg_norm', action='store_true')
    parser.add_argument('--scale_msg', action='store_true')
    parser.add_argument('--learn_msg_scale', action='store_true')

    return parser.parse_args()
