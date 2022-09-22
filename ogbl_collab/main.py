import hf_env
hf_env.set_env('202111')

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import hfai
hfai.set_watchdog_time(21600)
import hfai.nccl.distributed as dist
from hfai.datasets import OGB
from hfai.nn.parallel import DistributedDataParallel

from ogbl_collab.args import get_args
from ogbl_collab.eval import Evaluator
from ogbl_collab.model import DeeperGCN, LinkPredictor

SAVE_PATH = Path('./output/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def test(model, predictor, x, edge_index, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, edge_index)

    pos_train_edge = split_edge['train']['edge'].cuda(non_blocking=True)
    pos_valid_edge = split_edge['valid']['edge'].cuda(non_blocking=True)
    neg_valid_edge = split_edge['valid']['edge_neg'].cuda(non_blocking=True)
    pos_test_edge = split_edge['test']['edge'].cuda(non_blocking=True)
    neg_test_edge = split_edge['test']['edge_neg'].cuda(non_blocking=True)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    train_hits = evaluator.eval({
        'y_pred_pos': pos_train_pred,
        'y_pred_neg': neg_valid_pred,
    })[f'hits@50']
    valid_hits = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })[f'hits@50']
    test_hits = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })[f'hits@50']

    return train_hits, valid_hits, test_hits


def train(model, predictor, x, edge_index, split_edge, optimizer, batch_size):

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].cuda(non_blocking=True)

    total_loss, total_examples = 0, 0

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):

        optimizer.zero_grad()
        h = model(x, edge_index)
        # positive edges
        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        # add a extremely small value to avoid gradient explode
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # negative edges
        edge = torch.randint(0, x.size(0), edge.size(), dtype=torch.long, device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def main(local_rank):
    args = get_args()
    sub_dir = '{}-epochs_{}'.format(args.dataset, args.epochs)

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

    dataset = OGB(data_name=args.dataset)
    data = dataset.get_data()
    # Data(edge_index=[2, 2358104], edge_weight=[2358104, 1], edge_year=[2358104, 1], x=[235868, 128])
    split_edge = dataset.get_split()
    evaluator = Evaluator(args.dataset, metric='hits@50')

    x = data.x.cuda(non_blocking=True)
    edge_index = data.edge_index.cuda(non_blocking=True)

    args.in_channels = data.x.size(-1)
    args.num_tasks = 1

    model = DeeperGCN(args)
    predictor = LinkPredictor(args)
    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])
    predictor = DistributedDataParallel(predictor.cuda(), device_ids=[local_rank])
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)


    # load
    start_epoch, best_eval = 0, 0
    if (save_path / f'deepgcn_{args.dataset}_latest.pt').exists():
        ckpt = torch.load(save_path / f'deepgcn_{args.dataset}_latest.pt', map_location="cpu")
        model.module.load_state_dict(ckpt['model'])
        predictor.load_state_dict(ckpt['predictor'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt["epoch"]
        best_eval = ckpt["best_eval"]

    if local_rank == 0:
        print(f"Start training for {args.epochs} epochs", flush=True)

    for epoch in range(start_epoch, args.epochs):

        epoch_loss = train(model, predictor, x, edge_index, split_edge, optimizer, args.batch_size)
        train_hits, valid_hits, test_hits = test(model, predictor, x, edge_index, split_edge, evaluator, args.batch_size)

        if local_rank == 0:
            print(f"Epoch {epoch} | epoch_loss: {epoch_loss:.6f}, hits: {train_hits:.6f}, {valid_hits:.6f}, {test_hits:.6f}", flush=True)

            if valid_hits > best_eval:
                best_eval = valid_hits
                states = {
                    'model': model.module.state_dict(),
                    'predictor': model.module.state_dict()
                }
                torch.save(states, save_path / f'deepgcn_{args.dataset}_best.pt')

            states = {
                'model': model.module.state_dict(),
                'predictor': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_eval': best_eval,
                'epoch': epoch+1
            }
            torch.save(states, save_path / f'deepgcn_{args.dataset}_latest.pt')

        torch.cuda.synchronize()


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)
