import hf_env
hf_env.set_env('202111')

import os
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn

import hfai
hfai.set_watchdog_time(21600)
import hfai.nccl.distributed as dist
from hfai.nn.parallel import DistributedDataParallel

from ogbn_proteins.dataset import OGBNDataset
from ogbn_proteins.model import DeeperGCN
from ogbn_proteins.args import get_args
from ogbn_proteins.eval import Evaluator
from utils.data_util import intersection, process_indexes

SAVE_PATH = Path('./output/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def train(data, dataset, model, optimizer, criterion):

    loss_list = []
    model.train()
    sg_nodes, sg_edges, sg_edges_index, _ = data

    train_y = dataset.y[dataset.train_idx]
    idx_clusters = np.arange(len(sg_nodes))
    np.random.shuffle(idx_clusters)

    for idx in idx_clusters:

        x = dataset.x[sg_nodes[idx]].float().cuda(non_blocking=True)
        sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).cuda(non_blocking=True)

        sg_edges_ = sg_edges[idx].cuda(non_blocking=True)
        sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].cuda(non_blocking=True)

        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}

        inter_idx = intersection(sg_nodes[idx], dataset.train_idx.tolist())
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        optimizer.zero_grad()

        pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr)

        target = train_y[inter_idx].cuda(non_blocking=True)

        loss = criterion(pred[training_idx].to(torch.float32), target.to(torch.float32))
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        if dist.get_rank() == 0 and hfai.receive_suspend_command():
            hfai.go_suspend()

    return np.mean(loss_list)


@torch.no_grad()
def multi_evaluate(valid_data_list, dataset, model, evaluator):
    model.eval()
    target = dataset.y.detach().numpy()

    train_pre_ordered_list = []
    valid_pre_ordered_list = []
    test_pre_ordered_list = []

    test_idx = dataset.test_idx.tolist()
    train_idx = dataset.train_idx.tolist()
    valid_idx = dataset.valid_idx.tolist()

    for valid_data_item in valid_data_list:
        sg_nodes, sg_edges, sg_edges_index, _ = valid_data_item
        idx_clusters = np.arange(len(sg_nodes))

        test_predict = []
        train_predict = []
        valid_predict = []
        test_target_idx = []
        train_target_idx = []
        valid_target_idx = []

        for idx in idx_clusters:
            x = dataset.x[sg_nodes[idx]].float().cuda(non_blocking=True)
            sg_nodes_idx = torch.LongTensor(sg_nodes[idx]).cuda(non_blocking=True)

            sg_edges_ = sg_edges[idx].cuda(non_blocking=True)
            sg_edges_attr = dataset.edge_attr[sg_edges_index[idx]].cuda(non_blocking=True)

            inter_tr_idx = intersection(sg_nodes[idx], train_idx)
            inter_v_idx = intersection(sg_nodes[idx], valid_idx)
            inter_te_idx = intersection(sg_nodes[idx], test_idx)

            train_target_idx += inter_tr_idx
            valid_target_idx += inter_v_idx
            test_target_idx += inter_te_idx

            mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}
            tr_idx = [mapper[tr_idx] for tr_idx in inter_tr_idx]
            v_idx = [mapper[v_idx] for v_idx in inter_v_idx]
            te_idx = [mapper[te_idx] for te_idx in inter_te_idx]

            pred = model(x, sg_nodes_idx, sg_edges_, sg_edges_attr).cpu().detach()

            train_predict.append(pred[tr_idx])
            valid_predict.append(pred[v_idx])
            test_predict.append(pred[te_idx])

        train_pre = torch.cat(train_predict, 0).numpy()
        valid_pre = torch.cat(valid_predict, 0).numpy()
        test_pre = torch.cat(test_predict, 0).numpy()

        train_pre_ordered = train_pre[process_indexes(train_target_idx)]
        valid_pre_ordered = valid_pre[process_indexes(valid_target_idx)]
        test_pre_ordered = test_pre[process_indexes(test_target_idx)]

        train_pre_ordered_list.append(train_pre_ordered)
        valid_pre_ordered_list.append(valid_pre_ordered)
        test_pre_ordered_list.append(test_pre_ordered)

    train_pre_final = torch.mean(torch.Tensor(train_pre_ordered_list), dim=0)
    valid_pre_final = torch.mean(torch.Tensor(valid_pre_ordered_list), dim=0)
    test_pre_final = torch.mean(torch.Tensor(test_pre_ordered_list), dim=0)

    eval_result = {}

    input_dict = {"y_true": target[train_idx], "y_pred": train_pre_final}
    train_eval = torch.FloatTensor([evaluator.eval(input_dict)]).cuda()

    input_dict = {"y_true": target[valid_idx], "y_pred": valid_pre_final}
    valid_eval = torch.FloatTensor([evaluator.eval(input_dict)]).cuda()

    input_dict = {"y_true": target[test_idx], "y_pred": test_pre_final}
    test_eval = torch.FloatTensor([evaluator.eval(input_dict)]).cuda()

    total = torch.FloatTensor([1.0]).cuda()

    for x in [train_eval, valid_eval, test_eval, total]:
        dist.reduce(x, 0)

    if dist.get_rank() == 0:
        return {
            'train': train_eval.item() / total.item(),
            'valid': valid_eval.item() / total.item(),
            'test': test_eval.item() / total.item()
        }
    else:
        return {
            'train': train_eval.item(),
            'valid': valid_eval.item(),
            'test': test_eval.item()
        }


def main(local_rank):
    args = get_args()
    sub_dir = '{}-train_{}-test_{}-num_evals_{}-epochs_{}'.format(args.dataset, args.cluster_number, args.valid_cluster_number, args.num_evals, args.epochs)

    save_path = SAVE_PATH / sub_dir
    save_path.mkdir(parents=True, exist_ok=True)

    # init dist
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "54247")
    hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
    rank = int(os.environ.get("RANK", "0"))  # node id
    gpus = torch.cuda.device_count()  # gpus per node

    # fix the seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42 + rank * gpus + local_rank)
    cudnn.benchmark = True

    dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    dataset = OGBNDataset(dataset_name=args.dataset)
    # extract initial node features
    args.nf_path = dataset.extract_node_features(args.aggr)
    args.num_tasks = dataset.num_tasks

    evaluator = Evaluator(args.dataset, args.num_tasks, "rocauc")
    criterion = torch.nn.BCEWithLogitsLoss()

    valid_data_list = []
    for i in range(args.num_evals):
        parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.valid_cluster_number)
        valid_data = dataset.generate_sub_graphs(parts, cluster_number=args.valid_cluster_number)
        valid_data_list.append(valid_data)

    model = DeeperGCN(args)
    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * hosts * gpus)

    # load
    start_epoch, best_eval = 0, 0
    if (save_path / f'deepgcn_{args.dataset}_latest.pt').exists():
        ckpt = torch.load(save_path / f'deepgcn_{args.dataset}_latest.pt', map_location="cpu")
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt["epoch"]
        best_eval = ckpt["best_eval"]

    if local_rank == 0:
        print(f"Start training for {args.epochs} epochs", flush=True)

    for epoch in range(start_epoch, args.epochs // (hosts * gpus)):
        # do random partition every epoch
        train_parts = dataset.random_partition_graph(dataset.total_no_of_nodes, cluster_number=args.cluster_number)
        data = dataset.generate_sub_graphs(train_parts, cluster_number=args.cluster_number)

        epoch_loss = train(data, dataset, model, optimizer, criterion)

        result = multi_evaluate(valid_data_list, dataset, model, evaluator)

        if local_rank == 0:
            print(f"Epoch {epoch} | epoch_loss: {epoch_loss:.6f}, valid: {result}", flush=True)

            if result['valid'] > best_eval:
                torch.save(model.module.state_dict(), save_path / f'deepgcn_{args.dataset}_best.pt')
                best_eval = result['valid']

            states = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_eval': best_eval,
                'epoch': epoch+1
            }
            torch.save(states, save_path / f'deepgcn_{args.dataset}_latest.pt')

        torch.cuda.synchronize()



if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)
