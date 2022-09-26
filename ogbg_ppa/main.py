import hf_env
hf_env.set_env('202111')

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from functools import partial
from torch_geometric.loader import DataLoader

import hfai
hfai.set_watchdog_time(21600)
import hfai.nccl.distributed as dist
from hfai.nn.parallel import DistributedDataParallel

from ogbg_ppa.args import get_args
from ogbg_ppa.dataset import OGBGDataset
from ogbg_ppa.model import DeeperGCN
from ogbg_ppa.eval import Evaluator
from utils.data_util import extract_node_feature

SAVE_PATH = Path('./output/')
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def train(model, loader, optimizer, criterion):
    loss_list = []
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.cuda(non_blocking=True)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:

            optimizer.zero_grad()
            pred = model(batch)

            loss = criterion(pred.to(torch.float32), batch.y.view(-1, ))

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

    loss_val = torch.from_numpy(np.mean(loss_list)).cuda()
    dist.reduce(loss_val, 0)

    return loss_val.item() / dist.get_world_size()


@torch.no_grad()
def eval(model, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.cuda(non_blocking=True)

        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            y_true.append(batch.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    acc = evaluator.eval({"y_true": y_true, "y_pred": y_pred})['acc']
    acc = torch.from_numpy(acc).cuda()
    dist.reduce(acc, 0)

    return acc.item()


def main(local_rank):
    args = get_args()
    sub_dir = '{}-bs_{}-nf_{}'.format(args.dataset, args.batch_size, args.aggr)

    save_path = SAVE_PATH / sub_dir
    save_path.mkdir(parents=True, exist_ok=True)

    # fix the seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    cudnn.benchmark = True

    # init dist
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "54247")
    hosts = int(os.environ.get("WORLD_SIZE", "1"))  # number of nodes
    rank = int(os.environ.get("RANK", "0"))  # node id
    gpus = torch.cuda.device_count()  # gpus per node

    dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts * gpus, rank=rank * gpus + local_rank)
    torch.cuda.set_device(local_rank)

    extract_node_feature_func = partial(extract_node_feature, reduce=args.aggr)
    dataset = OGBGDataset(data_name='ogbg_ppa', transform=extract_node_feature_func)

    split_idx = dataset.split
    # train_data = dataset[split_idx["train"]]
    # valid_data = dataset[split_idx["valid"]]
    # test_data = dataset[split_idx["test"]]
    #
    # train_sampler = DistributedSampler(train_data)
    # valid_sampler = DistributedSampler(valid_data)
    # test_sampler = DistributedSampler(test_data)
    #
    # train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    # valid_loader = DataLoader(valid_data, batch_size=args.batch_size, sampler=valid_sampler, num_workers=8, pin_memory=True)
    # test_loader = DataLoader(test_data, batch_size=args.batch_size, sampler=test_sampler, num_workers=8, pin_memory=True)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)

    evaluator = Evaluator(args.dataset)

    model = DeeperGCN(args)
    model = DistributedDataParallel(model.cuda(), device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * hosts * gpus)
    criterion = torch.nn.CrossEntropyLoss()

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

    for epoch in range(start_epoch, args.epochs):

        epoch_loss = train(model, train_loader, optimizer, criterion)

        train_accuracy = eval(model, train_loader, evaluator)
        valid_accuracy = eval(model, valid_loader, evaluator)
        test_accuracy = eval(model, test_loader, evaluator)

        if local_rank == 0:
            print(f'Epoch {epoch} | Loss: {epoch_loss:.6f}, Train: {train_accuracy:.6f}, Validation: {valid_accuracy:.6f}, Test: {test_accuracy:.6f}')

            if valid_accuracy > best_eval:
                best_eval = valid_accuracy
                torch.save(model.module.state_dict(), save_path / f'deepgcn_{args.dataset}_best.pt')

            states = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_eval': best_eval,
                'epoch': epoch + 1
            }
            torch.save(states, save_path / f'deepgcn_{args.dataset}_latest.pt')

        torch.cuda.synchronize()


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)
